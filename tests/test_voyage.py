import math

import numpy as np
import pytest

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.evaluate import Sea, recost
from shiprouting.route.voyage import voyage
from shiprouting.weather.forecast import Forecast
from test_fuel_route import NAMES, T0
from test_plan import GRAPH, LAND

START, GOAL = (0.5, 150.5), (0.5, 156.5)


@pytest.fixture(autouse=True)
def coarse_tables(monkeypatch):
    build = waves.AddedResistanceTable.build.__func__
    monkeypatch.setattr(waves.AddedResistanceTable, "build", classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi))))


def steady(hours=200, **values) -> Forecast:
    lat, lon = np.arange(-90.0, 90.5, 1.0), np.arange(0.0, 360.0, 1.0)
    time = T0 + np.arange(0, hours + 1, 6) * np.timedelta64(3600, "s")
    return Forecast(lat, lon, time, {n: np.full((len(time), len(lat), len(lon)), values.get(n, 0.0)) for n in NAMES})


class Same:
    """Every run is the same forecast."""

    def __init__(self, forecast):
        self.forecast = forecast

    def newest(self, t):
        return T0, self.forecast


def test_a_voyage_replanned_in_the_weather_forecast_sails_as_planned():
    fc = steady(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0, current_u=0.3)
    deadline = T0 + np.timedelta64(45 * 3600, "s")
    out = voyage(GRAPH, KCS_SHIP, LAND, Same(fc), fc, START, GOAL, T0, deadline, via=[(4.5, 152.5)],
                 speeds_kn=(12, 16, 20), every_hours=8.0)
    sailed = out["sailed"]
    assert sailed["plans"] > 2
    assert (sailed["lat"][-1], sailed["lon"][-1]) == pytest.approx(GOAL)
    assert (4.5, 152.5) in set(zip(sailed["lat"], sailed["lon"]))  # the waypoint is passed once, before it is dropped
    assert sailed["late_h"] == 0 and not sailed["deadline_moved"]
    assert sailed["fuel_t"] == pytest.approx(out["planned"]["least_fuel_t"], rel=0.01)  # the weather came as forecast
    assert out["shortest"]["fuel_t"] == pytest.approx(out["planned"]["shortest_t"], rel=1e-3)
    once = voyage(GRAPH, KCS_SHIP, LAND, Same(fc), fc, START, GOAL, T0, deadline, via=[(4.5, 152.5)],
                  speeds_kn=(12, 16, 20), every_hours=math.inf)
    assert once["sailed"]["plans"] == 1
    assert once["sailed"]["fuel_t"] == pytest.approx(out["planned"]["least_fuel_t"], rel=1e-3)


def test_a_saved_voyage_sailed_again_in_the_same_weather_burns_the_same():
    fc = steady(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0, current_u=0.3)
    deadline = T0 + np.timedelta64(45 * 3600, "s")
    out = voyage(GRAPH, KCS_SHIP, LAND, Same(fc), fc, START, GOAL, T0, deadline, via=[(4.5, 152.5)],
                 speeds_kn=(12, 16, 20), every_hours=8.0)
    again = voyage(GRAPH, KCS_SHIP, LAND, Same(fc), fc, START, GOAL, T0, deadline, via=[(4.5, 152.5)],
                   speeds_kn=(12, 16, 20), every_hours=8.0)
    for name in ("shortest", "sailed"):
        assert again[name]["segment_kg"] == out[name]["segment_kg"]  # the same inputs give the same voyage
        same = recost(out, Sea(KCS_SHIP, fc, T0), (12, 16, 20))[name]
        assert same["fuel_t"] == pytest.approx(out[name]["fuel_t"], rel=1e-4)
        assert same["hours"] == pytest.approx(out[name]["hours"], rel=1e-4)
    calm = recost(out, Sea(KCS_SHIP, steady(), T0), (12, 16, 20))
    assert calm["sailed"]["fuel_t"] < out["sailed"]["fuel_t"]  # judged in other weather
    assert calm["sailed"]["lat"] == out["sailed"]["lat"]  # on the same track


def test_on_time_keeps_the_track_and_chooses_speeds_that_arrive_by_the_deadline():
    from shiprouting.route.evaluate import on_time, track

    sea = Sea(KCS_SHIP, steady(Hs=2.0, T_m10=8.0, wave_from_x=1.0, wind_u=-8.0), T0)
    lat, lon = np.zeros(7), np.linspace(150.0, 153.0, 7)
    on_link = np.zeros(6, bool)
    km = 3.0 * 111.195
    hours = km / (15.0 * 1.852)  # needs 15 kn on average; planned at 14 kn, it is late
    kg, h, sailed, _ = sea.sail(lat, lon, on_link, np.full(6, 14.0), (12, 14, 16), 0.0)
    saved = {"hours": hours, "sailed": track(lat, lon, on_link, np.full(6, 14.0), kg, h, sailed, hours)}
    assert saved["sailed"]["late_h"] > 0
    out = on_time(saved, sea, (12, 14, 16))["sailed"]
    assert out["lat"] == saved["sailed"]["lat"] and out["late_h"] == 0.0 and out["hours"] <= round(hours, 1)
    assert out["fuel_t"] > saved["sailed"]["fuel_t"]  # faster on average than 14 kn
    assert set(out["segment_kn"]) <= {12.0, 14.0, 16.0} and 16.0 in out["segment_kn"]
