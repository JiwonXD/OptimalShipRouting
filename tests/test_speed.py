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


def test_slack_is_spent_when_slowing_down_meets_a_stronger_current_later():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    fc = forecast(wind_u=-5.0, Hs=1.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["current_u"][1] = -3.0  # m/s: a head current growing over the voyage, so a later
    fc.time[1] = T0 + np.timedelta64(24, "h")  # segment takes longer than foreseen when it starts later
    sc = SpeedChoice([dataclasses.replace(c, currents=True) for c in choice(fc, (12, 14, 16, 18)).costs])
    s, g = graph.nearest(0.5, 150.5), graph.nearest(0.5, 155.5)
    fixed = sc.costs[2]  # 16 kn
    r16 = astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)
    eta = r16.clock[-1] * 1.06
    r = sc.fill_slack(dataclasses.replace(r16, cost=sc.path(r16)), eta)
    assert r16.clock[-1] < r.clock[-1] <= eta
    assert r.cost < sc.path(r16)
    assert r.cost == pytest.approx(sc.path(r))


def test_starting_from_a_price_that_met_the_eta_before_takes_fewer_tries():
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    sc = choice(fc, (12, 14, 16, 18, 20))
    s, g = graph.nearest(0.5, 150.5), graph.nearest(5.5, 160.5)
    fixed = sc.costs[2]  # 16 kn
    eta = astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0).clock[-1] * 1.001
    cold, cold_tries = meet_eta(graph, s, g, sc, eta)
    price = min(t["per_hour"] for t in cold_tries if t["in_time"])
    warm, warm_tries = meet_eta(graph, s, g, sc, eta, price=price * 1.02)  # the weather has moved on a little
    assert len(warm_tries) < len(cold_tries)
    assert warm.clock[-1] <= eta * (1 + 1e-9)
    assert warm.cost == pytest.approx(cold.cost, rel=0.005)


def test_retime_finds_the_least_fuel_speeds_on_a_fixed_way():
    import itertools

    from shiprouting.sea.latlon import SearchResult

    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    fc.fields["Hs"][1] = 5.0
    sc = choice(fc, (12, 16, 20))
    lat, lon = np.zeros(5), np.linspace(150.0, 152.0, 5)
    r = SearchResult(lat, lon, np.zeros(4, bool), [], 0.0, 0, 0.0, clock=np.zeros(5))
    eta = 7.5
    best = math.inf
    for k in itertools.product(range(3), repeat=4):  # every choice of speeds
        clock, kg, _ = sc.sail(r, np.array(k))
        if clock[-1] <= eta:
            best = min(best, kg[np.array(k), np.arange(4)].sum())
    out = sc.retime(r, eta, dt_hours=0.01)
    assert out.clock[-1] <= eta
    assert out.cost == pytest.approx(best, rel=1e-9)


def _varied_forecast():
    """A forecast whose fields change in place and in time over 4 steps, with a few corners missing."""
    from test_fuel_route import NAMES

    from shiprouting.weather.forecast import Forecast

    rng = np.random.default_rng(7)
    lat, lon = np.arange(-10.0, 10.5, 1.0), np.arange(140.0, 160.5, 1.0)
    times = T0 + np.arange(4) * np.timedelta64(6, "h")
    base = {"wind_u": (-6.0, 6.0), "wind_v": (-6.0, 6.0), "Hs": (0.5, 5.0), "wave_from_x": (-1.0, 1.0), "wave_from_y": (-1.0, 1.0),
            "T_m10": (5.0, 11.0), "current_u": (-1.0, 1.0), "current_v": (-1.0, 1.0)}  # fmt: skip
    fields = {n: rng.uniform(*base[n], (4, len(lat), len(lon))) for n in NAMES}
    for n in ("Hs", "current_u"):
        fields[n][rng.random(fields[n].shape) < 0.05] = np.nan
    return Forecast(lat, lon, times, fields)


def _varied_choice(currents):
    fc = _varied_forecast()
    return SpeedChoice([
        FuelCost(KCS_SHIP, fc, T0, kn * KNOT, AddedResistanceTable.build(KCS_SEAKEEPING, kn * KNOT, T01=(7.0, 9.0), angle=(0.0, math.pi / 2, math.pi)),
                 frozen=False, currents=currents)
        for kn in (12, 16, 20)
    ])  # fmt: skip


@pytest.mark.parametrize("currents", [False, True])
def test_one_move_at_many_times_matches_the_moves_one_by_one(currents):
    from shiprouting.route import kernel
    from shiprouting.route.fuel import stacked_tables

    sc = _varied_choice(currents)
    hours = np.linspace(0.0, 25.0, 301)  # past the forecast's end too
    kg, h = sc.kg_hours(np.full_like(hours, -3.3), np.full_like(hours, 147.2), np.full_like(hours, -2.6), np.full_like(hours, 148.1), hours)
    kg2, h2, counts = np.empty_like(kg), np.empty_like(h), np.zeros((3, 3), np.int64)
    kernel.fuel_kg_times(-3.3, 147.2, -2.6, 148.1, hours, kg2, h2, counts, *stacked_tables(sc.costs))
    np.testing.assert_allclose(kg2, kg, rtol=1e-12)
    np.testing.assert_allclose(h2, h, rtol=1e-12)


@pytest.mark.parametrize("currents", [False, True])
def test_moves_at_their_own_speed_match_every_speed_picked(currents):
    sc = _varied_choice(currents)
    rng = np.random.default_rng(3)
    lat_a, lon_a = rng.uniform(-9, 9, 500), rng.uniform(141, 159, 500)
    lat_b, lon_b = lat_a + rng.uniform(-0.5, 0.5, 500), lon_a + rng.uniform(-0.5, 0.5, 500)
    hours, at = rng.uniform(0.0, 25.0, 500), rng.integers(0, 3, 500)  # past the forecast's end too
    kg, h = sc.kg_hours(lat_a, lon_a, lat_b, lon_b, hours)
    kg1, h1 = sc.kg_hours_at(lat_a, lon_a, lat_b, lon_b, hours, at)
    assert np.array_equal(kg1, kg[at, np.arange(500)]) and np.array_equal(h1, h[at, np.arange(500)])


@pytest.mark.parametrize("currents", [False, True])
def test_relax_at_one_speed_per_segment_moves_the_points_as_before(currents, monkeypatch):
    from shiprouting.sea.latlon import SearchResult

    sc = _varied_choice(currents)
    lat = np.linspace(-6.0, 4.0, 41) + np.where(np.arange(41) % 2, 0.3, -0.3)
    lon = np.linspace(142.0, 157.0, 41)
    on_link = np.zeros(40, bool)
    on_link[20] = True
    k = np.random.default_rng(5).integers(0, 3, 40)
    r = SearchResult(lat, lon, on_link, [], 0.0, 0, 0.0, clock=np.zeros(41))
    r = dataclasses.replace(r, clock=sc.sail(r, k)[0])
    land = Land([shapely.box(150.0, 2.0, 151.0, 3.0)])
    out = sc.relax(land, r, keep_points=(10,))

    def every_speed_picked(self, lat_a, lon_a, lat_b, lon_b, hours, at):
        kg, h = self.kg_hours(lat_a, lon_a, lat_b, lon_b, hours)
        m = np.arange(kg.shape[1])
        return kg[at, m], h[at, m]

    monkeypatch.setattr(SpeedChoice, "kg_hours_at", every_speed_picked)
    ref = sc.relax(land, r, keep_points=(10,))
    assert not np.array_equal(ref.lat, r.lat)  # it did move points
    for name in ("lat", "lon", "clock"):
        assert np.array_equal(getattr(out, name), getattr(ref, name))
    assert out.cost == ref.cost


def _retime_reference(sc, r, eta_hours, dt_hours=0.02, max_current=2.5, per_hour=0.0):
    """SpeedChoice.retime as it was before its inner loop was compiled (kept to check against)."""
    from shiprouting.sea.latlon import haversine_km

    n = len(r.lat) - 1
    seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
    speed = np.array([c.speed for c in sc.costs])
    left_h = np.r_[np.cumsum(seg_km[::-1])[::-1], 0.0] / ((speed.max() + max_current) * 3.6)
    end = r.clock[0] + eta_hours * (1 + 1e-9)
    t, f, layers = np.array([r.clock[0]]), np.array([0.0]), []
    for i in range(n):
        kg, h = sc.kg_hours(r.lat[i], r.lon[i], r.lat[i + 1], r.lon[i + 1], t)
        t2, f2 = (t[None] + h).ravel(), (f[None] + kg).ravel()
        k2, p2 = np.repeat(np.arange(len(speed)), len(t)), np.tile(np.arange(len(t)), len(speed))
        ok = np.isfinite(f2) & (t2 + left_h[i + 1] <= end)
        t2, f2, k2, p2 = t2[ok], f2[ok], k2[ok], p2[ok]
        bucket, keep = np.floor(t2 / dt_hours), np.zeros(len(t2), bool)
        for value in (f2, f2 + per_hour * t2):
            order = np.lexsort((t2, value, bucket))
            keep[order[np.r_[True, bucket[order][1:] != bucket[order][:-1]]]] = True
        t, f = t2[keep], f2[keep]
        layers.append((p2[keep], k2[keep]))
    best, k = int(np.argmin(f)), np.empty(n, int)
    for i in range(n - 1, -1, -1):
        parent, ks = layers[i]
        k[i], best = ks[best], parent[best]
    return k


@pytest.mark.parametrize("currents", [False, True])
def test_retime_compiled_picks_the_same_speeds_as_before(currents):
    from shiprouting.sea.latlon import SearchResult

    sc = _varied_choice(currents)
    lat, lon = np.linspace(-6.0, 4.0, 41), np.linspace(142.0, 157.0, 41)
    r = SearchResult(lat, lon, np.zeros(40, bool), [], 0.0, 0, 0.0, clock=np.zeros(41))
    for eta, price in ((95.0, 0.0), (80.0, 3000.0), (70.0, 8000.0)):
        out = sc.retime(r, eta, dt_hours=0.05, per_hour=price)
        k = _retime_reference(sc, r, eta, dt_hours=0.05, per_hour=price)
        assert np.array_equal(sc.speed_index(out), k) or out.cost <= sc.sail(r, k)[1][k, np.arange(40)].sum() * (1 + 1e-12)
        _, kg, _ = sc.sail(r, k)
        assert out.cost == pytest.approx(kg[k, np.arange(40)].sum(), rel=1e-12)


@pytest.mark.parametrize("currents,price", [(False, 0.0), (True, 2500.0)])
def test_compiled_search_finds_what_astar_finds(currents, price):
    from shiprouting.sea.links import Link

    canal = Link("운하", np.array([0.0, 0.0]), np.array([148.6, 150.4]))
    graph = Graph.build(Land([shapely.box(149.0, -30.0, 150.0, 30.0)]), root=1.0, min_step=0.25, lat_limits=(-9.0, 9.0), links=[canal])
    sc = _varied_choice(currents)
    sc.per_hour = price
    s, g = graph.nearest(-4.5, 144.5), graph.nearest(5.5, 156.5)
    want = astar(graph, s, g, sc.edge_cost(graph), sc.link_cost, lambda n: 0.0, clock0=3.0)
    got = sc.search(graph, s, g, clock0=3.0, chunk=50, heap=16)  # resumed many times, its heap grown
    assert want.links == got.links == ["운하"]
    assert np.array_equal(got.lat, want.lat) and np.array_equal(got.lon, want.lon) and np.array_equal(got.on_link, want.on_link)
    assert np.array_equal(got.clock, want.clock)
    assert got.cost == want.cost and got.expanded == want.expanded


def test_shortcut_and_relax_judge_land_the_same_with_open_water_skipped():
    from shiprouting.sea.quadtree import open_water

    sc = _varied_choice(False)
    land = Land([shapely.box(147.0, -2.0, 148.0, 2.0), shapely.box(152.0, -3.0, 152.5, 3.0)])
    graph = Graph.build(land, root=1.0, min_step=0.25, lat_limits=(-20.0, 20.0))
    open_sea = open_water(graph, land)
    assert open_sea.safe.any() and not open_sea.safe.all()
    r = astar(graph, graph.nearest(-0.2, 143.0), graph.nearest(0.3, 157.0))  # round both islands
    k = np.random.default_rng(2).integers(0, 3, len(r.lat) - 1)
    r = dataclasses.replace(r, clock=sc.sail(r, k)[0])
    for way in (sc.shortcut, sc.relax):
        a, b = way(land, r, (), open_sea=open_sea), way(land, r, ())
        for name in ("lat", "lon", "clock", "on_link"):
            assert np.array_equal(getattr(a, name), getattr(b, name))
        assert a.cost == b.cost
