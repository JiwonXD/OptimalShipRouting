import dataclasses

import numpy as np
import pytest
import shapely

from shiprouting.sea.land import Land
from shiprouting.sea.latlon import LatLonGrid, haversine_km
from shiprouting.sea.latlon import astar as grid_astar
from shiprouting.sea.links import Link
from shiprouting.sea.quadtree import Graph, astar

# A wall of land from 30°S to 30°N with a strait 0.25° wide just north of the equator.
WALL = Land([shapely.box(0.0, -30.0, 2.0, 0.1), shapely.box(0.0, 0.35, 2.0, 30.0)])


@pytest.fixture(scope="module")
def graph():
    return Graph.build(WALL, root=1.0, min_step=1 / 16, lat_limits=(-40.0, 40.0))


def test_only_cells_along_the_coast_are_split(graph):
    sizes = dict(zip(*np.unique(graph.size, return_counts=True)))
    assert sizes[1.0] > 0.9 * 80 * 360  # the open ocean keeps its root cells
    assert min(sizes) == 1 / 16


def test_a_strait_narrower_than_the_root_cells_opens(graph):
    start, goal = (0.2, -3.0), (0.2, 5.0)
    coarse = LatLonGrid.build(WALL, 1.0, lat_limits=(-1.0, 1.0))
    r = grid_astar(coarse, (-0.5, -2.5), (-0.5, 4.5))
    assert r.km > 30_000  # on the regular 1° grid the strait is land: round the world instead ...
    r = astar(graph, graph.nearest(*start), graph.nearest(*goal))  # ... here it is open
    assert r.km < 1.2 * haversine_km(*start, *goal)
    inside = (r.lon > 0) & (r.lon < 2)
    assert np.all((r.lat[inside] > 0.1) & (r.lat[inside] < 0.35))


def test_cells_join_across_the_dateline(graph):
    r = astar(graph, graph.nearest(-10.5, 179.5), graph.nearest(-10.5, -179.5))
    assert r.km == pytest.approx(haversine_km(-10.5, 179.5, -10.5, -179.5))


def test_links_join_cells_near_their_ends(graph):
    canal = Link("운하", np.array([-20.0, -20.0]), np.array([-0.5, 2.5]))
    g = Graph.build(WALL, root=1.0, min_step=1 / 16, lat_limits=(-40.0, 40.0), links=[canal])
    r = astar(g, g.nearest(-20.5, -5.5), g.nearest(-20.5, 7.5))
    assert r.links == ["운하"]
    # The clock (km by default) runs along the path, through the canal's own points too.
    assert np.all(np.diff(r.clock) > 0)
    assert r.clock[-1] == pytest.approx(r.km)


def test_pockets_of_water_cut_off_from_the_open_sea_are_dropped():
    # A lagoon inside a ring of land is split into small water cells that connect only to each other.
    ring = shapely.box(0.1, 0.1, 0.9, 0.9).difference(shapely.box(0.3, 0.3, 0.7, 0.7))
    g = Graph.build(Land([ring]), root=1.0, min_step=1 / 8, lat_limits=(-10.0, 10.0))
    in_lagoon = (g.lat > 0.3) & (g.lat < 0.7) & (g.lon > 0.3) & (g.lon < 0.7)
    assert not in_lagoon.any()
    assert g.size.min() == 1 / 8  # the coast outside the ring is still split


def test_straightening_turns_a_staircase_into_a_straight_line(graph):
    from shiprouting.sea.quadtree import straighten

    start, goal = (-20.5, -60.5), (-12.5, -30.5)  # open water, not along a row, column or diagonal
    r = astar(graph, graph.nearest(*start), graph.nearest(*goal))
    t = straighten(WALL, r)
    assert len(t.lat) == 2
    assert t.km == pytest.approx(haversine_km(*start, *goal))
    assert t.km < r.km


def test_straightened_legs_go_round_land(graph):
    from shiprouting.sea.quadtree import arcs_on_land, straighten

    r = astar(graph, graph.nearest(-35.5, -5.5), graph.nearest(25.5, 6.5))  # round the south end of the wall
    t = straighten(WALL, r)
    assert not arcs_on_land(WALL, t.lat[:-1], t.lon[:-1], t.lat[1:], t.lon[1:]).any()
    assert 2 < len(t.lat) < len(r.lat)
    assert t.km < r.km


def test_straightening_keeps_links_and_crosses_the_dateline():
    from shiprouting.sea.quadtree import straighten

    canal = Link("운하", np.array([-20.0, -20.0]), np.array([-0.5, 2.5]))
    g = Graph.build(WALL, root=1.0, min_step=1 / 16, lat_limits=(-40.0, 40.0), links=[canal])
    t = straighten(WALL, astar(g, g.nearest(-25.5, -5.5), g.nearest(-15.5, 7.5)))
    assert t.links == ["운하"] and t.on_link.sum() == 1
    assert [-20.0, -0.5] in np.column_stack([t.lat, t.lon]).tolist()
    t = straighten(WALL, astar(g, g.nearest(-10.5, 175.5), g.nearest(-5.5, -170.5)))
    assert len(t.lat) == 2


def test_arcs_are_checked_along_the_great_circle_not_the_lon_lat_line():
    from shiprouting.sea.latlon import segments_on_land
    from shiprouting.sea.quadtree import arcs_on_land, along

    # The great circle between two points on 45.5°N, 60° of longitude apart, bulges north
    # halfway to tan(lat) = tan(45.5°) / cos(30°), about 49.6°N.
    mid_lat, mid_lon = along(45.5, -60.5, 45.5, -0.5, 0.5)
    assert mid_lat == pytest.approx(np.degrees(np.arctan(np.tan(np.radians(45.5)) / np.cos(np.radians(30)))))
    assert mid_lon == pytest.approx(-30.5)
    island = Land([shapely.box(-31.0, 48.5, -30.0, 50.5)])
    assert segments_on_land(island, np.array([45.5, 45.5]), np.array([-60.5, -0.5])) == 0
    assert arcs_on_land(island, 45.5, -60.5, 45.5, -0.5).all()
    assert not arcs_on_land(island, 45.5, -60.5, 45.5, -35.5).any()


def test_arcs_across_the_dateline_see_land_on_both_sides():
    from shiprouting.sea.quadtree import arcs_on_land

    land = Land([shapely.box(-180.0, -1.0, -179.9, 1.0), shapely.box(179.0, 10.0, 180.0, 11.0)])
    assert arcs_on_land(land, [0.0, 10.5], [179.5, 178.5], [0.0, 10.5], [-179.5, -178.5]).tolist() == [True, True]
    assert not arcs_on_land(land, 5.0, 179.5, 5.0, -179.5).any()


def test_arcs_can_be_required_to_keep_clear_of_land():
    from shiprouting.sea.quadtree import arcs_on_land

    island = Land([shapely.box(10.0, 0.0, 10.1, 0.1)])
    # A leg along 0.105°N passes about 0.56 km north of the island.
    assert not arcs_on_land(island, 0.105, 9.0, 0.105, 11.0).any()
    assert not arcs_on_land(island, 0.105, 9.0, 0.105, 11.0, clearance_km=0.3).any()
    assert arcs_on_land(island, 0.105, 9.0, 0.105, 11.0, clearance_km=1.0).all()


def path(points, on_link=None):
    from shiprouting.sea.latlon import SearchResult

    lat, lon = (np.array(v, float) for v in zip(*points))
    on_link = np.zeros(len(lat) - 1, bool) if on_link is None else np.array(on_link)
    return SearchResult(lat, lon, on_link, [], float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()), 0, 0.0)


def test_a_spike_is_smoothed_out():
    from shiprouting.sea.latlon import turn_deg
    from shiprouting.sea.quadtree import smooth_turns

    r = path([(0, 0), (0, 1), (0.3, 1.05), (0, 1.1), (0, 2)])  # a 0.3° detour north and straight back
    s = smooth_turns(Land([]), r)
    assert (turn_deg(s.lat, s.lon) < 60).all()
    assert s.lat.tolist() == [0, 0, 0, 0] and s.lon == pytest.approx([0, 1, 1.1, 2])
    assert s.km < r.km


def test_a_links_turns_are_kept():
    from shiprouting.sea.quadtree import smooth_turns

    r = path([(0, 0), (0, 1), (1, 1.1), (0, 1.2), (0, 2)], on_link=[False, True, True, False])  # the spike is a link's
    assert smooth_turns(Land([]), r).lat.tolist() == r.lat.tolist()


def test_a_corner_that_cannot_be_dropped_is_cut():
    from shiprouting.sea.latlon import turn_deg
    from shiprouting.sea.quadtree import arcs_on_land, smooth_turns

    corner = Land([shapely.box(0.2, 0.2, 2.0, 2.0)])
    r = path([(0, 1), (0, 0.1), (1, 0.1)])  # 90° round the corner of the land
    r = dataclasses.replace(r, clock=np.array([0.0, 10.0, 20.0]))
    s = smooth_turns(corner, r)
    assert (turn_deg(s.lat, s.lon) < 60).all()
    assert not arcs_on_land(corner, s.lat[:-1], s.lon[:-1], s.lat[1:], s.lon[1:]).any()
    assert [s.lat[0], s.lon[0], s.lat[-1], s.lon[-1]] == pytest.approx([0, 1, 1, 0.1]) and s.km < r.km
    assert len(s.clock) == len(s.lat) and (np.diff(s.clock) > 0).all() and s.clock[[0, -1]].tolist() == [0, 20]


def test_a_v_shaped_corner_is_cut_twice():
    from shiprouting.sea.latlon import turn_deg
    from shiprouting.sea.quadtree import smooth_turns

    wall = Land([shapely.box(0.99, 0.3, 1.01, 2.0)])  # the corner can't be dropped: the wall is in the way
    r = path([(1, 0.9), (0, 1), (1, 1.1)])  # turns by about 168°
    s = smooth_turns(wall, r)
    assert len(s.lat) >= 5
    assert (turn_deg(s.lat, s.lon) < 60).all()
