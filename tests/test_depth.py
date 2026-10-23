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
