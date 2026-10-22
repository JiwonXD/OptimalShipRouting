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
