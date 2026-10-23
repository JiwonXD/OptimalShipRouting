import math

import numpy as np
import pytest
import shapely

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route import plan as plan_module
from shiprouting.sea.land import Land
from shiprouting.sea.quadtree import Graph
from test_fuel_route import T0, forecast

GRAPH = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
START, GOAL = (0.5, 150.5), (0.5, 156.5)


@pytest.fixture(autouse=True)
def coarse_tables(monkeypatch):
    build = waves.AddedResistanceTable.build.__func__
    monkeypatch.setattr(waves.AddedResistanceTable, "build", classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi))))


def plan(deadline_hours, speeds=(12, 16, 20)):
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["wind_u"][1] = 10.0
    return plan_module.plan(GRAPH, KCS_SHIP, fc, START, GOAL, T0, T0 + np.timedelta64(int(deadline_hours * 3600), "s"), speeds)


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
