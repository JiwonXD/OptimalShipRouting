import numpy as np
import pytest
import shapely

from shiprouting.sea.land import Land
from shiprouting.sea.latlon import LatLonGrid, astar, haversine_km, segments_on_land
from shiprouting.sea.links import Link, chain


def way(nodes, coords):
    return {"nodes": nodes, "geometry": [{"lat": la, "lon": lo} for la, lo in coords]}


def test_chain_follows_shared_nodes_between_the_nearest_ends():
    ways = [
        way([1, 2, 3], [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]),
        way([3, 4], [(0.0, 2.0), (1.0, 2.0)]),
        way([2, 5], [(0.0, 1.0), (-1.0, 1.0)]),  # a side branch the chain must not take
    ]
    link = chain("운하", ways, (0.1, -0.1), (1.1, 2.0))
    assert link.lat.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert link.lon.tolist() == [0.0, 1.0, 2.0, 2.0]
    assert link.km == pytest.approx(2 * haversine_km(0, 0, 0, 1) + haversine_km(0, 2, 1, 2))


def test_chain_fails_when_the_ways_do_not_connect():
    ways = [way([1, 2], [(0.0, 0.0), (0.0, 1.0)]), way([3, 4], [(0.0, 2.0), (0.0, 3.0)])]
    with pytest.raises(ValueError):
        chain("운하", ways, (0.0, 0.0), (0.0, 3.0))


def test_route_takes_a_link_through_a_wall_of_land():
    land = Land([shapely.box(-1.0, -60.0, 1.0, 60.0)])
    canal = Link("운하", np.array([10.0, 10.0, 10.0]), np.array([-1.5, 0.0, 1.5]))
    grid = LatLonGrid.build(land, 1.0, lat_limits=(-90.0, 90.0), links=[canal])
    r = astar(grid, (10.5, -5.5), (10.5, 5.5))
    assert r.links == ["운하"]
    assert r.km < 2 * haversine_km(10.5, -5.5, 10.5, 5.5)  # instead of round the end of the wall
    assert r.on_link.sum() == 2  # the two segments of the canal itself
    assert segments_on_land(land, r.lat, r.lon) == 2  # both canal segments cross the wall ...
    assert segments_on_land(land, r.lat, r.lon, skip=r.on_link) == 0  # ... and nothing else does
