import math

import numpy as np
import pytest
import shapely

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route import plan as plan_module
from shiprouting.sea.land import Land
from shiprouting.sea.latlon import haversine_km, turn_deg
from shiprouting.sea.quadtree import Graph
from test_fuel_route import T0, forecast

LAND = Land([shapely.box(100.0, 30.0, 101.0, 31.0)])
GRAPH = Graph.build(LAND, root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
START, GOAL = (0.5, 150.5), (0.5, 156.5)


@pytest.fixture(autouse=True)
def coarse_tables(monkeypatch):
    build = waves.AddedResistanceTable.build.__func__
    monkeypatch.setattr(waves.AddedResistanceTable, "build", classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi))))


def plan(deadline_hours, speeds=(12, 16, 20), via=(), legs="distance", land=None):
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["wind_u"][1] = 10.0
    return plan_module.plan(GRAPH, KCS_SHIP, fc, START, GOAL, T0, T0 + np.timedelta64(int(deadline_hours * 3600), "s"), speeds,
                            via=via, legs=legs, land=land)


def test_both_routes_arrive_by_the_deadline_and_the_least_fuel_one_takes_no_more():
    out = plan(25.0)
    short, best = out["shortest"], out["least_fuel"]
    assert short["hours"] == pytest.approx(25.0, abs=0.1)  # the shortest route sails at the speed that just meets it
    assert best["hours"] <= 25.0 + 0.05
    assert best["fuel_t"] <= short["fuel_t"]
    assert out["saving_t"] == pytest.approx(short["fuel_t"] - best["fuel_t"], abs=0.01)


def test_a_deadline_too_close_is_reported():
    out = plan(10.0)
    assert "cannot be met" in out["error"] and "shortest" not in out


def test_a_far_deadline_sails_the_shortest_route_at_the_slowest_speed():
    out = plan(200.0)
    assert out["shortest"]["speeds_kn"] == {"12": len(out["shortest"]["lat"]) - 1}
    assert out["shortest"]["hours"] < 200.0


@pytest.mark.parametrize("legs", ["distance", "shared"])
@pytest.mark.parametrize("land", [None, LAND])
def test_both_routes_pass_the_waypoints_in_order_and_arrive_by_the_deadline(legs, land):
    via = [(4.5, 152.5), (-3.5, 154.5)]
    out = plan(70.0, via=via, legs=legs, land=land)
    for key in ("shortest", "least_fuel"):
        r = out[key]
        at = [int(np.argmin(np.hypot(np.subtract(r["lat"], p[0]), np.subtract(r["lon"], p[1])))) for p in via]
        assert at[0] < at[1]
        assert [(r["lat"][k], r["lon"][k]) for k in at] == [pytest.approx(p) for p in via]
    assert out["shortest"]["hours"] == pytest.approx(70.0, abs=0.1)
    assert out["least_fuel"]["hours"] <= 70.0 + 0.05
    assert out["least_fuel"]["fuel_t"] <= out["shortest"]["fuel_t"]
    assert out["shortest"]["km"] > plan(70.0)["shortest"]["km"] + 500  # the detour is sailed


def test_smoothing_leaves_sharp_turns_only_at_the_waypoints():
    via = [(4.5, 152.5), (-3.5, 154.5)]  # the course turns by about 100° at each
    out = plan(70.0, via=via, legs="shared", land=LAND)
    for key in ("shortest", "least_fuel"):
        r = out[key]
        turn = np.r_[0.0, turn_deg(np.array(r["lat"]), np.array(r["lon"])), 0.0]
        assert {(r["lat"][k], r["lon"][k]) for k in np.nonzero(turn >= 60)[0]} == set(via)
    legs = out["least_fuel"]["legs"]
    assert len(legs) == 3 and sum(leg["hours"] for leg in legs) == pytest.approx(out["least_fuel"]["hours"], abs=0.2)


def test_against_a_current_the_shortest_route_speeds_up_to_keep_the_deadline():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0, current_u=-1.0)
    deadline = T0 + np.timedelta64(30 * 3600, "s")
    out = plan_module.plan(GRAPH, KCS_SHIP, fc, START, GOAL, T0, deadline, (12, 16, 20))
    still = plan_module.plan(GRAPH, KCS_SHIP, fc, START, GOAL, T0, deadline, (12, 16, 20), currents=False)
    assert 30.0 * (1 - 0.002) <= out["shortest"]["hours"] <= 30.0  # just before the deadline, never after
    assert float(next(iter(out["shortest"]["speeds_kn"]))) > float(next(iter(still["shortest"]["speeds_kn"]))) + 1.5
    assert out["least_fuel"]["hours"] <= 30.0 + 0.05
    assert out["least_fuel"]["fuel_t"] <= out["shortest"]["fuel_t"]


def test_the_shortest_route_follows_the_great_circle_where_the_sea_is_open():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    start, goal = (0.5, 150.5), (4.5, 165.5)  # not a grid direction: a few diagonal steps, then straight along a row
    out = plan_module.plan(GRAPH, KCS_SHIP, fc, start, goal, T0, T0 + np.timedelta64(80, "h"), (12, 16, 20), land=LAND,
                           least_fuel=False)
    assert out["shortest"]["km"] == pytest.approx(float(haversine_km(*start, *goal)), rel=0.002)


def test_the_least_fuel_route_cuts_the_grid_steps_in_calm_open_sea():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    start, goal = (0.5, 150.5), (4.5, 165.5)
    out = plan_module.plan(GRAPH, KCS_SHIP, fc, start, goal, T0, T0 + np.timedelta64(80, "h"), (12, 16, 20), land=LAND)
    assert out["least_fuel"]["km"] == pytest.approx(float(haversine_km(*start, *goal)), rel=0.01)
    assert out["least_fuel"]["fuel_t"] <= out["shortest"]["fuel_t"]


def test_the_least_fuel_route_keeps_going_round_a_storm():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    lat, lon = np.arange(-90.0, 90.5, 1.0), np.arange(0.0, 360.0, 1.0)
    storm = (np.abs(lat - 0.5) <= 2)[:, None] & ((lon >= 155) & (lon <= 160))[None, :]
    for name, value in (("Hs", 7.0), ("wind_u", -30.0)):
        fc.fields[name][:, storm] = value
    out = plan_module.plan(GRAPH, KCS_SHIP, fc, START, (0.5, 165.5), T0, T0 + np.timedelta64(60, "h"), (12, 16, 20), land=LAND)
    r = out["least_fuel"]
    lat, lon = np.array(r["lat"]), np.array(r["lon"])
    assert not ((np.abs(lat - 0.5) < 2.5) & (lon > 155) & (lon < 160)).any()  # round it, not through
    assert r["fuel_t"] < out["shortest"]["fuel_t"]
