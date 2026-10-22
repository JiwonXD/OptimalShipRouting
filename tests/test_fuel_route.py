import math

import numpy as np
import pytest
import shapely

from shiprouting.fuel.ships import KCS_SEAKEEPING, KCS_SHIP
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost, bearing
from shiprouting.sea.land import Land
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import Graph, astar
from shiprouting.weather.conditions import T01_PER_TM10, sea_state, weather_at
from shiprouting.weather.forecast import Forecast

NAMES = ("wind_u", "wind_v", "Hs", "wave_from_x", "wave_from_y", "T_m10", "current_u", "current_v")
T0 = np.datetime64("2026-10-21T18:00", "s")


def forecast(**values) -> Forecast:
    """A global 1° forecast with each field constant (0 unless given)."""
    lat, lon = np.arange(-90.0, 90.5, 1.0), np.arange(0.0, 360.0, 1.0)
    fields = {n: np.full((2, len(lat), len(lon)), values.get(n, 0.0)) for n in NAMES}
    return Forecast(lat, lon, np.array([T0, T0 + np.timedelta64(6, "h")]), fields)


# A coarse table: the full one takes about a minute to build.
WAVES = AddedResistanceTable.build(KCS_SEAKEEPING, 16 * KNOT, T01=(7.0, 9.0), angle=(0.0, math.pi / 2, math.pi))


def cost(fc) -> FuelCost:
    return FuelCost(KCS_SHIP, fc, T0, 16 * KNOT, WAVES)


def test_bearing():
    assert bearing(0, 0, 0, 1) == pytest.approx(math.pi / 2)
    assert bearing(0, 0, 1, 0) == pytest.approx(0.0)
    assert bearing(0, 1, 0, 0) == pytest.approx(-math.pi / 2)


def test_still_weather_costs_the_calm_water_fuel():
    c = cost(forecast())
    kg = c.kg(10.0, 150.0, 10.5, 150.5)
    assert kg == pytest.approx(c.calm_kg_per_km * haversine_km(10.0, 150.0, 10.5, 150.5), rel=1e-4)
    assert (c.no_waves, c.over_mcr) == (0, 0)


def test_head_wind_and_waves_cost_more_than_following():
    # Wind and waves from the east.
    c = cost(forecast(wind_u=-15.0, Hs=4.0, wave_from_x=1.0, T_m10=9.0))
    east, west = c.kg([10.0, 10.0], [150.0, 150.5], [10.0, 10.0], [150.5, 150.0])
    assert east > west > 0


def test_moves_cost_what_the_ship_model_gives_on_a_wave_table_point():
    # Heading east on the equator into wind from the north-east and waves from the east (T01 7 s).
    fc = forecast(wind_u=-8.0, wind_v=-6.0, Hs=3.0, wave_from_x=1.0, T_m10=7.0 / T01_PER_TM10)
    sea = sea_state(weather_at(fc, 0.0, 150.25, T0), math.pi / 2)
    hours = haversine_km(0.0, 150.0, 0.0, 150.5) / (16 * KNOT * 3.6)
    assert cost(fc).kg(0.0, 150.0, 0.0, 150.5) == pytest.approx(KCS_SHIP.fuel_rate(16 * KNOT, sea) * hours, rel=1e-4)


def test_moves_beyond_the_engine_cost_infinity():
    c = cost(forecast(Hs=20.0, wave_from_x=1.0, T_m10=9.0))
    assert c.kg(0.0, 150.0, 0.0, 150.5) == math.inf
    assert c.over_mcr == 1


def test_missing_waves_fall_back_to_wind_only():
    c = cost(forecast(wind_u=-15.0, Hs=math.nan))
    wind_only = cost(forecast(wind_u=-15.0)).kg(10.0, 150.0, 10.0, 150.5)
    assert c.kg(10.0, 150.0, 10.0, 150.5) == pytest.approx(wind_only)
    assert c.no_waves == 1


@pytest.fixture(scope="module")
def graph():
    return Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))


def test_fuel_search_in_still_weather_follows_the_shortest_route(graph):
    s, g = graph.nearest(0.5, 150.5), graph.nearest(5.5, 160.5)
    c = cost(forecast())
    r = astar(graph, s, g, c.edge_costs(graph, chunk=10_000), c.link_cost, c.heuristic(graph, g))
    shortest = astar(graph, s, g)
    assert r.km == pytest.approx(shortest.km)
    assert r.cost == pytest.approx(c.calm_kg_per_km * r.km, rel=1e-4)
    assert r.cost == pytest.approx(c.path(r.lat, r.lon, r.on_link))
