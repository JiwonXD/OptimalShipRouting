import math

import numpy as np
import pytest
import shapely

from shiprouting.fuel.ships import KCS_SEAKEEPING, KCS_SHIP
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.speed import SpeedChoice, meet_eta
from shiprouting.sea.land import Land
from shiprouting.sea.quadtree import Graph, astar
from test_fuel_route import T0, forecast


def choice(fc, speeds_kn):
    return SpeedChoice([
        FuelCost(KCS_SHIP, fc, T0, kn * KNOT, AddedResistanceTable.build(KCS_SEAKEEPING, kn * KNOT, T01=(7.0, 9.0), angle=(0.0, math.pi / 2, math.pi)), frozen=False)
        for kn in speeds_kn
    ])  # fmt: skip


def test_without_an_arrival_time_the_slowest_speed_wins():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    sc = choice(forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0), (12, 16, 20))
    s, g = graph.nearest(0.5, 150.5), graph.nearest(5.5, 160.5)
    r = astar(graph, s, g, sc.edge_cost(graph), sc.link_cost, lambda n: 0.0)
    assert np.allclose(sc.speeds_kn(r), 12.0)
    assert r.clock[-1] == pytest.approx(r.km / (12 * 1.852))
    assert r.cost == pytest.approx(sc.path(r))


def test_meeting_an_arrival_time_takes_less_fuel_than_a_fixed_speed():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["wind_u"][1] = 10.0  # the head wind turns into a following wind
    sc = choice(fc, (12, 14, 16, 18, 20))
    s, g = graph.nearest(0.5, 150.5), graph.nearest(0.5, 156.5)
    fixed = sc.costs[2]  # 16 kn
    r16 = astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)
    r, tries = meet_eta(graph, s, g, sc, r16.clock[-1], tolerance=0.01)
    assert r.clock[-1] <= r16.clock[-1]
    assert r.cost < r16.cost
    assert r.cost == pytest.approx(sc.path(r))
    assert len(tries) > 1
