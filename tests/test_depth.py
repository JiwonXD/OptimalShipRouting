import numpy as np
import pytest

from shiprouting.sea.land import Land


@pytest.fixture
def gebco(tmp_path):
    """A 1° GEBCO-like grid: 0.25° cells, 40 m deep everywhere but a 10 m bank around (0.5°N, 0.5°E)."""
    h5py = pytest.importorskip("h5py")
    step = 0.25
    lat = np.arange(-1 + step / 2, 2, step)
    lon = np.arange(-180 + step / 2, 180, step)
    z = np.full((len(lat), len(lon)), -40, np.int16)
    z[np.ix_(np.abs(lat - 0.5) < 0.2, np.abs(lon - 0.5) < 0.2)] = -10
    z[:, -1] = -5  # the last column, just west of the dateline
    path = tmp_path / "gebco.nc"
    with h5py.File(path, "w") as f:
        f["lat"], f["lon"], f["elevation"] = lat, lon, z
    return path


def test_elevation_is_read_off_the_cell_holding_the_point(gebco):
    from shiprouting.sea.depth import Bathymetry

    b = Bathymetry(gebco)
    assert b.elevation([0.5, 0.0, 0.5], [0.5, 0.0, 179.99]).tolist() == [-10, -40, -5]
    assert b.elevation(0.5, -180.0 + 1e-9) == -40 and b.elevation(0.5, 180.0 - 1e-9) == -5


def test_legs_over_a_bank_are_found_and_links_are_left_out(gebco):
    from shiprouting.sea.depth import Bathymetry, summary

    b = Bathymetry(gebco)
    s = summary(b, [0.5, 0.5, 1.5], [0.0, 1.0, 1.0])
    assert s["shallow_legs"] == 1 and s["least_m"] == 10 and s["where"][0][2] == 10
    assert (s["start_m"], s["goal_m"]) == (40, 40)
    assert summary(b, [0.5, 0.5, 1.5], [0.0, 1.0, 1.0], on_link=[True, False])["shallow_legs"] == 0


def test_cells_over_shallow_water_are_left_out_of_the_graph(gebco):
    from shiprouting.sea.depth import Bathymetry
    from shiprouting.sea.quadtree import Graph

    graph = Graph.build(Land([]), lat_limits=(0, 1), bathymetry=Bathymetry(gebco))
    deep = Graph.build(Land([]), lat_limits=(0, 1))
    assert len(deep.lat) - len(graph.lat) == 4 + 4  # the bank and the column by the dateline
    assert not ((np.abs(graph.lat - 0.5) < 0.2) & (np.abs(graph.lon - 0.5) < 0.2)).any()


def test_a_line_over_a_bank_counts_as_crossing_land(gebco):
    import shapely

    from shiprouting.sea.depth import Bathymetry, LandOrShallow

    obstacles = LandOrShallow(Land([]), Bathymetry(gebco), chunk=50)
    lines = shapely.linestrings([[[0.0, 0.5], [1.0, 0.5]], [[0.0, 1.5], [1.0, 1.5]], [[179.0, 0.5], [181.0, 0.5]]])
    assert obstacles.crosses(lines).tolist() == [True, False, True]  # the last one runs past the dateline into 5 m
    assert obstacles.contains([0.5, 1.5], [0.5, 0.5]).tolist() == [True, False]


def test_moves_over_a_bank_are_left_out(gebco):
    from shiprouting.sea.depth import Bathymetry, LandOrShallow
    from shiprouting.sea.quadtree import Graph, astar

    graph = Graph.build(LandOrShallow(Land([]), Bathymetry(gebco)), lat_limits=(0, 1))
    r = astar(graph, graph.nearest(0.5, 0.0), graph.nearest(0.5, 1.0))
    assert not ((r.lat > 0.25) & (r.lat < 0.75) & (r.lon > 0.25) & (r.lon < 0.75)).any()


def test_water_near_an_exempt_point_is_not_shallow(gebco):
    from shiprouting.sea.depth import Bathymetry, LandOrShallow, summary

    b = Bathymetry(gebco)
    near = LandOrShallow(Land([]), b, exempt=[(0.5, 0.5)], exempt_km=30)
    assert near.contains([0.5, 0.5], [0.5, 179.9]).tolist() == [False, True]
    s = summary(b, [0.5, 0.5], [0.0, 1.0], exempt=[(0.5, 0.5)], exempt_km=30)
    assert s["shallow_legs"] == 0 and s["least_m"] == 40


class Bank:
    """A GEBCO stand-in on 0.025° cells: 100 m deep, but 10 m over a bank at 144–146°E, cut by a 0.05°
    wide channel 30 m deep just north of the equator."""

    step = 0.025
    lat0, lon0 = -90 + step / 2, -180 + step / 2

    def elevation(self, lat, lon):
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        bank = (lon >= 144) & (lon <= 146)
        return np.where(bank & (lat >= 0) & (lat < 0.05), -30, np.where(bank, -10, -100))


def test_a_channel_too_narrow_for_the_grid_is_traced_as_a_link():
    import shapely

    from shiprouting.sea.depth import channel
    from shiprouting.sea.quadtree import Graph, astar

    land = Land([shapely.box(-180.0, -90.0, 140.0, 90.0), shapely.box(150.0, -90.0, 180.0, 90.0)])
    a, b = (0.5, 143.0), (0.5, 147.0)
    cut = Graph.build(land, root=1.0, min_step=0.5, lat_limits=(-5.0, 5.0), bathymetry=Bank())
    with pytest.raises(ValueError, match="no route"):  # every cell over the bank is 10 m deep at its centre
        astar(cut, cut.nearest(*a), cut.nearest(*b))

    link = channel(Bank(), land, "해협", a, b, min_depth=15.0)
    assert (link.lat[0], link.lon[0]) == pytest.approx(a, abs=0.02) and (link.lat[-1], link.lon[-1]) == pytest.approx(b, abs=0.02)
    f = np.linspace(0, 1, 200)[:, None]  # every leg of the link stays over the channel
    lat = (link.lat[:-1] + f * (link.lat[1:] - link.lat[:-1])).ravel()
    lon = (link.lon[:-1] + f * (link.lon[1:] - link.lon[:-1])).ravel()
    assert (-Bank().elevation(lat, lon) > 15).all()
    assert len(link.lat) < 10  # cut down to the corners

    joined = Graph.build(land, root=1.0, min_step=0.5, lat_limits=(-5.0, 5.0), bathymetry=Bank(), links=[link])
    r = astar(joined, joined.nearest(*a), joined.nearest(*b))
    assert r.links == ["해협"]
