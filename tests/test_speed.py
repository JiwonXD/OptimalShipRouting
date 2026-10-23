import dataclasses
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


def test_meeting_a_fixed_speeds_arrival_time_takes_no_more_fuel():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["wind_u"][1] = 10.0  # the head wind turns into a following wind
    sc = choice(fc, (12, 14, 16, 18, 20))
    s, g = graph.nearest(0.5, 150.5), graph.nearest(0.5, 156.5)
    fixed = sc.costs[2]  # 16 kn
    r16 = astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)
    r, tries = meet_eta(graph, s, g, sc, r16.clock[-1], tolerance=0.01)
    assert r.clock[-1] <= r16.clock[-1]
    assert r.cost <= r16.cost * (1 + 1e-9)  # here the best mix at that time is 16 kn throughout
    assert r.cost == pytest.approx(sc.path(r))
    assert len(tries) > 1


def test_time_left_before_the_eta_is_spent_slowing_down():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    sc = choice(forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0), (12, 14, 16, 18))
    s, g = graph.nearest(0.5, 150.5), graph.nearest(0.5, 156.5)
    fixed = sc.costs[2]  # 16 kn
    r16 = astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)
    eta = r16.clock[-1] * 1.1
    r = sc.fill_slack(r16, eta)
    assert r16.clock[-1] < r.clock[-1] <= eta
    assert r.cost < r16.cost
    assert r.cost == pytest.approx(sc.path(r))
    assert set(np.round(sc.speeds_kn(r), 6)) == {14.0, 16.0}


def test_all_speeds_in_one_pass_match_each_speed_alone():
    fc = forecast(wind_u=-10.0, wind_v=3.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["Hs"][1] = 4.0
    sc = choice(fc, (12, 16, 20))
    lat_a, lon_a, hours = np.array([0.5, 10.0, -5.0]), np.array([150.5, 160.0, 170.0]), np.array([0.0, 3.0, 9.0])
    args = (lat_a, lon_a, lat_a + 0.5, lon_a + 0.5, hours)
    together = sc.kg(*args)
    assert together.shape == (3, 3)
    for c, kg in zip(sc.costs, together):
        assert kg == pytest.approx(dataclasses.replace(c, beyond_forecast=0).kg(*args))
    assert [c.beyond_forecast for c in sc.costs] == [1, 1, 1]


def test_a_band_of_head_current_is_sailed_around_and_the_speeds_are_read_back():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    fc = forecast(wind_u=-5.0, Hs=1.0, wave_from_x=1.0, T_m10=8.0)
    band = (fc.lat >= 0) & (fc.lat <= 1)
    fc.fields["current_u"][:, band] = -4.0  # m/s, against a ship heading east
    s, g = graph.nearest(0.5, 150.5), graph.nearest(0.5, 158.5)
    still = choice(fc, (12, 16, 20))
    sc = SpeedChoice([dataclasses.replace(c, currents=True) for c in still.costs])
    straight = astar(graph, s, g, still.edge_cost(graph), still.link_cost, lambda n: 0.0)
    assert np.abs(straight.lat - 0.5).max() < 1
    r, _ = meet_eta(graph, s, g, sc, straight.clock[-1] * 1.05, tolerance=0.01)
    assert np.abs(r.lat - 0.5).max() > 1.5  # leaves the band
    assert r.clock[-1] <= straight.clock[-1] * 1.05 * (1 + 1e-9)
    assert set(np.round(sc.speeds_kn(r), 6)) <= {12.0, 16.0, 20.0}
    assert r.cost == pytest.approx(sc.path(r))
    assert sc.snap_up(r).clock == pytest.approx(r.clock)
