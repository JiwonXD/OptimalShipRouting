import pytest
import shapely

from shiprouting.sea.land import Land
from shiprouting.sea.latlon import MOVES, LatLonGrid, astar, haversine_km

LAND = Land([shapely.box(0.0, 0.0, 20.0, 20.0)])


@pytest.fixture(scope="module")
def grid():
    return LatLonGrid.build(LAND, 1.0, lat_limits=(-90.0, 90.0))


def test_grid_marks_cells_with_centres_on_land(grid):
    assert grid.water.shape == (180, 360)
    assert not grid.water[grid.cell(10.0, 10.0)]
    assert grid.water[grid.cell(10.0, 25.0)]
    assert grid.water.size - grid.water.sum() == 20 * 20


def test_straight_line_in_open_water(grid):
    r = astar(grid, (-30.5, -60.5), (-30.5, -40.5))
    assert r.lat[0] == -30.5 and r.lon[-1] == -40.5
    assert len(r.lat) == 21  # one cell per step along the parallel
    assert r.km == pytest.approx(20 * haversine_km(-30.5, 0.0, -30.5, 1.0))


def test_goes_around_land(grid):
    r = astar(grid, (10.5, -5.5), (10.5, 25.5))
    inside = (r.lat > 0) & (r.lat < 20) & (r.lon > 0) & (r.lon < 20)
    assert not inside.any()
    assert r.km > haversine_km(10.5, -5.5, 10.5, 25.5)


def test_start_on_land_is_rejected(grid):
    with pytest.raises(ValueError):
        astar(grid, (10.5, 10.5), (10.5, 25.5))


def test_route_across_the_dateline(grid):
    r = astar(grid, (-10.5, 175.5), (-10.5, -175.5))
    assert len(r.lat) == 10  # straight across ±180°, not around the world
    assert r.km == pytest.approx(9 * haversine_km(-10.5, 0.0, -10.5, 1.0))


def test_land_check_across_the_dateline():
    import numpy as np

    from shiprouting.sea.latlon import segments_on_land

    land = Land([shapely.box(-180.0, -1.0, -179.8, 1.0)])
    assert segments_on_land(land, np.array([0.0, 0.0]), np.array([179.9, -179.9])) == 1
    assert segments_on_land(land, np.array([5.0, 5.0]), np.array([179.9, -179.9])) == 0


def test_latitudes_outside_the_ice_limit_are_blocked():
    g = LatLonGrid.build(LAND, 1.0)
    assert not g.water[g.cell(80.5, 0.5)] and not g.water[g.cell(-70.5, 0.5)]
    assert g.water[g.cell(60.5, 0.5)]


def test_diagonal_moves_do_not_cut_land_corners(grid):
    # Going round the corner from (-0.5, 0.5) south of the land to (0.5, -0.5) west of it
    # must step through (-0.5, -0.5) instead of jumping diagonally over the land corner.
    r = astar(grid, (-0.5, 0.5), (0.5, -0.5))
    assert list(zip(r.lat, r.lon)) == [(-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]


def test_moves_across_land_between_two_water_cells_are_blocked():
    # A thin island between the centres at 9.5°E and 10.5°E: both cells are water.
    land = Land([shapely.box(9.9, -5.0, 10.1, 5.0)])
    g = LatLonGrid.build(land, 1.0)
    assert g.water[g.cell(0.5, 9.5)] and g.water[g.cell(0.5, 10.5)]
    r = astar(g, (0.5, 9.5), (0.5, 10.5))
    assert r.lat.max() > 5.0 or r.lat.min() < -5.0  # around the island's end


def test_diagonal_strait_between_two_land_cells_stays_open():
    # Land covers the centres north-west and south-east of the move, but the line misses it.
    land = Land([shapely.box(0.1, 1.05, 0.9, 1.9), shapely.box(1.05, 0.1, 1.9, 0.9)])
    g = LatLonGrid.build(land, 1.0)
    assert not g.water[g.cell(1.5, 0.5)] and not g.water[g.cell(0.5, 1.5)]
    assert g.moves[g.cell(0.5, 0.5)][MOVES.index((1, 1))]
    assert g.moves[g.cell(1.5, 1.5)][MOVES.index((-1, -1))]


def test_moves_wrap_and_see_land_across_the_dateline():
    land = Land([shapely.box(-180.0, -5.0, -179.9, 5.0)])
    g = LatLonGrid.build(land, 1.0)
    assert not g.moves[g.cell(0.5, 179.5)][MOVES.index((0, 1))]
    assert g.moves[g.cell(10.5, 179.5)][MOVES.index((0, 1))]
    assert not g.moves[g.cell(0.5, -179.5)][MOVES.index((0, -1))]
