import dataclasses
import math

import numpy as np
import pytest
import shapely
from scipy.integrate import solve_ivp

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.evaluate import Sea
from shiprouting.route.fuel import KNOT
from shiprouting.route.plan import plan
from shiprouting.route.zermelo import T0, least_time, shear
from shiprouting.sea.land import Land
from shiprouting.sea.quadtree import Graph

V = 16 * KNOT


@pytest.fixture(autouse=True)
def coarse_tables(monkeypatch):
    build = waves.AddedResistanceTable.build.__func__
    monkeypatch.setattr(waves.AddedResistanceTable, "build", classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi))))


def test_the_least_time_track_ends_where_it_should():
    s, x = 3e-5, 400e3
    T, north = least_time(x, V, s)
    assert least_time(x, V, 0.0)[0] == pytest.approx(x / V)
    assert T < x / V  # bending north into the current pays

    def zermelo(t, z):  # integrated on its own from the starting heading
        return [V * math.cos(z[2]) + s * z[1], V * math.sin(z[2]), -s * math.cos(z[2]) ** 2]

    end = solve_ivp(zermelo, (0, T), [0.0, 0.0, math.atan(s * T / 2)], rtol=1e-11, atol=1e-6, dense_output=True)
    assert end.y[0, -1] == pytest.approx(x, abs=0.01)
    assert end.y[1, -1] == pytest.approx(0.0, abs=0.01)
    assert end.sol(T / 2)[1] == pytest.approx(north, rel=1e-6)


def test_the_planner_finds_the_least_time_track_in_a_shear_current():
    land = Land([shapely.box(100.0, 30.0, 101.0, 31.0)])
    graph = Graph.build(land, root=0.5, min_step=0.5, lat_limits=(-6.0, 6.0))
    start, goal = (-0.25, 149.75), (-0.25, 153.75)
    x = math.radians(goal[1] - start[1]) * 6371.0088e3 * math.cos(math.radians(start[0]))
    T, _ = least_time(x, V, 3e-5)
    ship = dataclasses.replace(KCS_SHIP, windage=None)
    fc = shear(start[0], 3e-5)
    out = plan(graph, ship, fc, start, goal, T0, T0 + np.timedelta64(int(3 * T), "s"), (16,), land=land, keep_shortest=False)
    f = out["least_fuel"]
    _, h, _, _ = Sea(ship, fc, T0).sail(np.array(f["lat"]), np.array(f["lon"]), np.array(f["on_link"], bool),
                                        np.array(f["segment_kn"]), (16,), 0.0)
    assert h.sum() * 3600 == pytest.approx(T, rel=0.01)  # the pass line: within 1%
    assert h.sum() * 3600 >= T * (1 - 1e-3)  # and never faster than the exact answer
