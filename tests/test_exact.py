import math

import numpy as np
import pytest

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.exact import least_fuel_by
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.speed import SpeedChoice, meet_eta
from test_fuel_route import T0, forecast
from test_plan import GRAPH


@pytest.fixture(autouse=True)
def coarse_tables(monkeypatch):
    build = waves.AddedResistanceTable.build.__func__
    monkeypatch.setattr(waves.AddedResistanceTable, "build", classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi))))


def choice(fc):
    return SpeedChoice([FuelCost(KCS_SHIP, fc, T0, k * KNOT, frozen=False, currents=True) for k in (12, 16, 20)])


def test_the_exact_search_meets_the_deadline_with_no_more_fuel_than_the_price_search():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["wind_u"][1] = 10.0
    fc.fields["Hs"][1] = 5.0
    a, b = GRAPH.nearest(0.5, 150.5), GRAPH.nearest(0.5, 154.5)
    for eta in (16.0, 20.0, 26.0):
        c = choice(fc)
        r, _ = meet_eta(GRAPH, a, b, c, eta)
        ex = least_fuel_by(GRAPH, a, b, c, eta)
        assert ex.clock[-1] <= eta * (1 + 1e-9)
        assert ex.cost <= c.path(r) * (1 + 1e-9)
        assert ex.cost == pytest.approx(c.path(ex), rel=1e-6)  # its fuel is that of the path it gives


def test_no_path_in_time_is_said_so():
    a, b = GRAPH.nearest(0.5, 150.5), GRAPH.nearest(0.5, 154.5)
    with pytest.raises(ValueError):
        least_fuel_by(GRAPH, a, b, choice(forecast()), 5.0)


def test_in_a_calm_sea_the_exact_fuel_is_the_slowest_speed_that_arrives():
    fc = forecast()
    a, b = GRAPH.nearest(0.5, 150.5), GRAPH.nearest(0.5, 154.5)
    c = choice(fc)
    ex = least_fuel_by(GRAPH, a, b, c, 100.0)
    assert set(np.round(c.speeds_kn(ex), 6)) == {12.0}


def test_a_known_path_bounds_the_search():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    a, b = GRAPH.nearest(0.5, 150.5), GRAPH.nearest(0.5, 154.5)
    c = choice(fc)
    best = least_fuel_by(GRAPH, a, b, c, 20.0)
    bounded = least_fuel_by(GRAPH, a, b, c, 20.0, upper_kg=best.cost * 1.01)
    assert bounded.cost == pytest.approx(best.cost) and bounded.expanded <= best.expanded
    with pytest.raises(ValueError):
        least_fuel_by(GRAPH, a, b, c, 20.0, upper_kg=best.cost * 0.99)
