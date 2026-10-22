import pytest

from shiprouting.fuel.engine import Engine
from shiprouting.fuel.ships import KCS_ENGINE, KCS_SHIP

KNOT = 1852 / 3600


def test_sfoc_passes_through_published_points():
    for load, sfoc in KCS_ENGINE.sfoc_points:
        assert KCS_ENGINE.sfoc(load * KCS_ENGINE.mcr) == pytest.approx(sfoc, rel=1e-12)


def test_fuel_rate_units():
    engine = Engine(mcr=20e6, sfoc_points=((0.5, 170.0), (0.75, 170.0), (1.0, 170.0)))
    # 10 MW × 170 g/kWh = 1,700 kg/h
    assert engine.fuel_rate(10e6) == pytest.approx(1700.0)


def test_rejects_load_above_mcr():
    with pytest.raises(ValueError):
        KCS_ENGINE.sfoc(1.01 * KCS_ENGINE.mcr)


def test_kcs_service_speed_power_and_load():
    P_B = KCS_SHIP.brake_power(24 * KNOT)
    assert P_B == pytest.approx(28.57e6, rel=2e-3)
    # Sizing rule: 15 % sea margin on top still fits under 90 % MCR.
    assert 1.15 * P_B < 0.9 * KCS_ENGINE.mcr


def test_kcs_fuel_trends():
    speeds = [v * KNOT for v in range(12, 25)]
    rates = [KCS_SHIP.fuel_rate(v) for v in speeds]
    assert rates == sorted(rates)
    # Fuel per distance grows with speed: slowing down saves fuel per voyage.
    per_metre = [r / v for r, v in zip(rates, speeds)]
    assert per_metre == sorted(per_metre)
