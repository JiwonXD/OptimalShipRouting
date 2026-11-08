import numpy as np
import pytest

from shiprouting.fuel import uncertainty
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.fuel import KNOT, FuelCost
from test_fuel_route import T0, WAVES, forecast


def test_the_nominal_sample_is_the_ship_itself():
    ship = uncertainty.ship_with(KCS_SHIP, uncertainty.NOMINAL)
    for kn in (12, 16, 20):
        assert ship.fuel_rate(kn * KNOT) == pytest.approx(KCS_SHIP.fuel_rate(kn * KNOT), rel=1e-12)


def test_a_track_judged_on_its_resistance_parts_burns_what_kg_hours_gives():
    fc = forecast(wind_u=-12.0, wind_v=3.0, Hs=3.5, wave_from_x=1.0, T_m10=8.0, current_u=0.4)
    c = FuelCost(KCS_SHIP, fc, T0, 16 * KNOT, WAVES, currents=True)
    lat_a, lon_a = np.zeros(4), np.array([150.0, 150.5, 151.0, 151.5])
    lat_b, lon_b = np.array([0.5, 0.5, -0.5, 0.0]), lon_a + 0.5
    kg, _ = c.kg_hours(lat_a, lon_a, lat_b, lon_b)
    air, wave, h = c.resistance_parts(lat_a, lon_a, lat_b, lon_b)
    tables = uncertainty.rate_tables(KCS_SHIP, [16 * KNOT])
    fuel, over = uncertainty.track_fuel(np.zeros(4, int), air, wave, h, tables, 1.0, 1.0)
    assert fuel == pytest.approx(kg.sum(), rel=1e-3) and over == 0
    more, _ = uncertainty.track_fuel(np.zeros(4, int), air, wave, h, tables, 1.0, 1.3)
    heavy = uncertainty.rate_tables(uncertainty.ship_with(KCS_SHIP, [1.1, 0.18, 0.209, 1.008, 1.0, 1.0, 0.0]), [16 * KNOT])
    assert more > fuel and uncertainty.track_fuel(np.zeros(4, int), air, wave, h, heavy, 1.0, 1.0)[0] > fuel


def test_sobol_indices_of_a_linear_function():
    # Y = x1 + 2 x2 + 0 x3 with x uniform: S1 = ST = (1/5, 4/5, 0).
    rng = np.random.default_rng(1)
    n, d = 20000, 3
    A, B = rng.random((n, d)), rng.random((n, d))
    AB = np.repeat(A[None], d, axis=0)
    for i in range(d):
        AB[i, :, i] = B[:, i]
    a = np.array([1.0, 2.0, 0.0])
    S1, ST = uncertainty.sobol(A @ a, B @ a, AB @ a)
    assert S1 == pytest.approx([0.2, 0.8, 0.0], abs=0.03)
    assert ST == pytest.approx([0.2, 0.8, 0.0], abs=0.03)


def test_sobol_first_order_indices_stay_tight_far_from_zero():
    # Y = 20 + x1 + 2 x2: the same indices, but uncentred the S1 estimate scattered by about 1 at n = 512.
    rng = np.random.default_rng(3)
    n, d, a = 512, 3, np.array([1.0, 2.0, 0.0])
    S1s = []
    for _ in range(50):
        A, B = rng.random((n, d)), rng.random((n, d))
        AB = np.repeat(A[None], d, axis=0)
        for i in range(d):
            AB[i, :, i] = B[:, i]
        S1s.append(uncertainty.sobol(20 + A @ a, 20 + B @ a, 20 + AB @ a)[0])
    assert np.std(S1s, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    low, high = uncertainty.sobol_interval(20 + A @ a, 20 + B @ a, 20 + AB @ a, rng, rounds=200)[0]
    assert np.all(low <= [0.2, 0.8, 0.0]) and np.all(high >= [0.2, 0.8, 0.0])


def test_saltelli_samples_lie_in_the_ranges():
    A, B, AB = uncertainty.saltelli(64, np.random.default_rng(0))
    low, high = np.array([p.low for p in uncertainty.PARAMETERS]), np.array([p.high for p in uncertainty.PARAMETERS])
    for x in (A, B, AB.reshape(-1, len(low))):
        assert np.all((x >= low) & (x <= high))
    assert np.array_equal(AB[2][:, 2], B[:, 2]) and np.array_equal(AB[2][:, 3], A[:, 3])
