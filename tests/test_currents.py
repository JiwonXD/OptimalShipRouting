import math

import numpy as np
import pytest

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.currents import effect, over_ground
from test_fuel_route import T0, forecast

LAT, LON = [0.0, 0.0, 0.0], [150.0, 151.0, 152.0]  # due east


@pytest.fixture(autouse=True)
def coarse_tables(monkeypatch):
    build = waves.AddedResistanceTable.build.__func__
    monkeypatch.setattr(waves.AddedResistanceTable, "build", classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi))))


def test_current_along_the_track_adds_to_the_speed_and_across_takes_from_it():
    kn, on_link = [16.0, 16.0], [False, True]
    sog, c_along = over_ground(forecast(current_u=-2 * 1852 / 3600), LAT, LON, kn, on_link, T0)  # 2 kn from ahead
    assert sog == pytest.approx([14.0, 16.0])  # links have no current
    assert c_along == pytest.approx([-2.0, 0.0])
    sog, _ = over_ground(forecast(current_v=3 * 1852 / 3600), LAT, LON, kn, [False, False], T0)  # 3 kn across
    assert sog == pytest.approx([math.sqrt(16**2 - 9)] * 2)


def test_fuel_scales_with_the_hours_over_the_ground():
    found = {"lat": LAT, "lon": LON, "on_link": [False, False], "segment_kn": [16.0, 16.0]}
    still = effect(forecast(), KCS_SHIP, found, T0)
    assert still["hours_current"] == still["hours"] and still["fuel_current_t"] == still["fuel_t"]
    against = effect(forecast(current_u=-2 * 1852 / 3600), KCS_SHIP, found, T0)
    assert against["hours_current"] == pytest.approx(still["hours"] * 16 / 14, abs=0.1)
    assert against["fuel_current_t"] == pytest.approx(still["fuel_t"] * 16 / 14, rel=0.01)
    assert against["mean_along_kn"] == pytest.approx(-2.0)
