import math

import pytest

from shiprouting.fuel import wind
from shiprouting.fuel.ships import KCS_WINDAGE


def test_relative_wind_in_still_air_is_head_wind_at_ship_speed():
    V, angle = wind.relative_wind(0.0, 0.0, 10.0)
    assert V == pytest.approx(10.0)
    assert angle == pytest.approx(0.0)


def test_relative_wind_head_following_and_beam():
    assert wind.relative_wind(5.0, 0.0, 10.0) == pytest.approx((15.0, 0.0))
    # Following wind slower than the ship still feels like a head wind.
    assert wind.relative_wind(4.0, math.pi, 10.0) == pytest.approx((6.0, 0.0))
    V, angle = wind.relative_wind(15.0, math.pi, 10.0)
    assert V == pytest.approx(5.0)
    assert abs(angle) == pytest.approx(math.pi)
    V, angle = wind.relative_wind(10.0, math.pi / 2, 10.0)
    assert V == pytest.approx(10.0 * math.sqrt(2))
    assert angle == pytest.approx(math.pi / 4)


def test_coefficient_is_symmetric_and_hits_table_points():
    for deg, c in zip(KCS_WINDAGE.angles, KCS_WINDAGE.C_DA):
        a = math.radians(deg)
        assert KCS_WINDAGE.coefficient(a) == pytest.approx(c)
        assert KCS_WINDAGE.coefficient(-a) == pytest.approx(c)
    assert KCS_WINDAGE.coefficient(math.radians(7.5)) == pytest.approx((0.67 + 0.76) / 2)


def test_air_resistance_still_air_and_tail_wind():
    R = wind.air_resistance(KCS_WINDAGE, 0.0, 0.0, 12.0)
    assert R == pytest.approx(0.5 * wind.RHO_AIR * 0.67 * KCS_WINDAGE.A_XV * 144.0)
    # A tail wind faster than the ship pushes it forward.
    assert wind.air_resistance(KCS_WINDAGE, 20.0, math.pi, 12.0) < 0
