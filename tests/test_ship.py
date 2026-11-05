import math

import pytest

from shiprouting.fuel.ship import SeaState, WaveSystem
from shiprouting.fuel.ships import KCS_SHIP, KCS_WINDAGE
from shiprouting.fuel.wind import air_resistance

KNOT = 1852 / 3600
V = 18 * KNOT


def test_calm_sea_is_calm_water_plus_still_air():
    expected = KCS_SHIP.resistance(V) + air_resistance(KCS_WINDAGE, 0.0, 0.0, V)
    assert KCS_SHIP.total_resistance(V) == pytest.approx(expected)


def test_head_weather_costs_more_than_following_weather():
    def sea(angle):
        return SeaState(wind_speed=12.0, wind_angle=angle, waves=(WaveSystem(3.0, 7.0, angle),))

    calm = KCS_SHIP.fuel_rate(V)
    head = KCS_SHIP.fuel_rate(V, sea(0.0))
    following = KCS_SHIP.fuel_rate(V, sea(math.pi))
    assert calm < following < head


def test_swell_adds_to_wind_sea():
    wind_sea = WaveSystem(2.0, 6.0, 0.0)
    swell = WaveSystem(2.0, 10.0, 0.5, s=75.0)
    both = KCS_SHIP.total_resistance(V, SeaState(waves=(wind_sea, swell)))
    each = (
        KCS_SHIP.total_resistance(V, SeaState(waves=(wind_sea,)))
        + KCS_SHIP.total_resistance(V, SeaState(waves=(swell,)))
        - KCS_SHIP.total_resistance(V)
    )
    assert both == pytest.approx(each)
