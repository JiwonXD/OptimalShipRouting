import pytest
import shapely

from shiprouting.sea.land import Land
from shiprouting.sea.latlon import LatLonGrid, astar, haversine_km

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
