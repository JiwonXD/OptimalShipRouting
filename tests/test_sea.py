import numpy as np
import pytest
import shapely

from shiprouting.sea import grid
from shiprouting.sea.geometry import great_circle_points, haversine_km, path_geometry
from shiprouting.sea.land import Land
from shiprouting.sea.route import Router, length_km

# A square continent, and a wall of land from 30°S to 30°N cut by a 0.5° (≈55 km) east-west channel.
LAND = Land([
    shapely.box(0.0, 0.0, 20.0, 20.0),
    shapely.box(40.0, 0.25, 60.0, 30.0),
    shapely.box(40.0, -30.0, 60.0, -0.25),
])


@pytest.fixture(scope="module")
def graph():
    return grid.build(LAND, ocean_res=2, max_res=5)


@pytest.fixture(scope="module")
def router(graph):
    return Router(graph, LAND)


def test_haversine_quarter_meridian():
    assert haversine_km(0.0, 0.0, 90.0, 0.0) == pytest.approx(np.pi / 2 * 6371.0088)


def test_line_across_the_dateline_hits_land_on_the_far_side():
    land = Land([shapely.box(-179.5, -1.0, -179.0, 1.0)])
    lat, lon = great_circle_points(0.0, 179.0, 0.0, -178.0, 10.0)
    assert land.crosses([path_geometry(lat, lon)])[0]
    lat, lon = great_circle_points(0.0, 170.0, 0.0, 175.0, 10.0)
    assert not land.crosses([path_geometry(lat, lon)])[0]


def test_fraction_of_a_cell_on_the_dateline():
    cell = grid.h3.latlng_to_cell(0.0, 180.0, 2)
    (poly,) = grid.cell_polygons([cell])
    half = Land([shapely.box(0.0, -90.0, 180.0, 90.0)])  # eastern hemisphere
    assert 0.2 < half.fraction([poly])[0] < 0.8
    assert Land([shapely.box(-180.0, -90.0, 180.0, 90.0)]).fraction([poly])[0] == pytest.approx(1.0)


def test_no_node_on_land_and_no_edge_across_land(graph):
    assert not LAND.contains(graph.lat, graph.lon).any()
    lines = [path_geometry(graph.lat[[a, b]], graph.lon[[a, b]]) for a, b in zip(graph.edge_i, graph.edge_j)]
    assert not LAND.crosses(lines).any()
    # Small cells near the coast, large ones in the open ocean.
    res = np.array([grid.h3.get_resolution(int(c)) for c in graph.cells])
    assert res.min() == 2 and res.max() == 5


def test_route_goes_around_land_and_is_straightened(router):
    start, end = (10.0, -10.0), (10.0, 30.0)
    lat, lon = router.route(start, end)
    assert (lat[0], lon[0]) == start and (lat[-1], lon[-1]) == end
    for k in range(len(lat) - 1):
        assert router.clear(lat[k], lon[k], lat[k + 1], lon[k + 1])
    direct = haversine_km(*start, *end)
    # Around the 20° square: longer than direct, but not by a detour of the whole continent.
    assert direct < length_km(lat, lon) < 1.5 * direct
    assert len(lat) <= 4


def test_route_uses_the_narrow_channel(router):
    lat, lon = router.route((0.0, 35.0), (0.0, 65.0))
    assert length_km(lat, lon) < 1.1 * haversine_km(0.0, 35.0, 0.0, 65.0)


def test_route_across_the_dateline_is_short(router):
    lat, lon = router.route((-10.0, 170.0), (-10.0, -170.0))
    assert length_km(lat, lon) == pytest.approx(haversine_km(-10.0, 170.0, -10.0, -170.0), rel=0.01)


def test_save_and_load(tmp_path, graph):
    graph.save(tmp_path / "g.npz")
    again = grid.SeaGraph.load(tmp_path / "g.npz")
    assert np.array_equal(again.cells, graph.cells)
    assert np.array_equal(again.edge_km, graph.edge_km)


def test_load_land_polygons_from_shapefile(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    from shiprouting.sea.land import load_osm_land_polygons

    wkb = shapely.to_wkb(np.array([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)], dtype=object))
    pyogrio.raw.write(tmp_path / "land.shp", wkb, field_data=[np.array([1, 2])], fields=["FID"],
                      geometry_type="Polygon", crs="EPSG:4326")
    land = load_osm_land_polygons(tmp_path / "land.shp")
    assert land.contains([0.5, 1.5], [0.5, 1.5]).tolist() == [True, False]


def test_canal_through_land(tmp_path, graph):
    import json

    from shiprouting.sea.canals import from_overpass

    # Two ways crossing the square continent from west to east along 10°N, plus a dead-end stub.
    ways = [
        {"type": "way", "nodes": [1, 2, 3], "geometry": [
            {"lat": 10.0, "lon": -0.5}, {"lat": 10.0, "lon": 5.0}, {"lat": 10.0, "lon": 10.0}]},
        {"type": "way", "nodes": [3, 4, 5], "geometry": [
            {"lat": 10.0, "lon": 10.0}, {"lat": 10.0, "lon": 15.0}, {"lat": 10.0, "lon": 20.5}]},
        {"type": "way", "nodes": [3, 6], "geometry": [{"lat": 10.0, "lon": 10.0}, {"lat": 12.0, "lon": 10.0}]},
    ]
    path = tmp_path / "canal.json"
    path.write_text(json.dumps({"elements": ways}))
    canal = from_overpass("Test", path, ends=((10.0, -0.5), (10.0, 20.5)))
    assert canal.lon.tolist() == [-0.5, 5.0, 10.0, 15.0, 20.5]
    assert canal.length_km == pytest.approx(haversine_km(10.0, -0.5, 10.0, 20.5), rel=1e-3)

    start, end = (10.0, -5.0), (10.0, 25.0)
    without = Router(graph, LAND).route(start, end)
    lat, lon = Router(graph, LAND, canals=(canal,)).route(start, end)
    assert length_km(lat, lon) < length_km(*without)
    assert 5.0 in lon.tolist() and 15.0 in lon.tolist()  # follows the centreline through the land
