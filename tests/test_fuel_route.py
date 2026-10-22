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
    r = astar(graph, s, g, c.costs(graph), c.link_cost, lambda n: 0.0)
    shortest = astar(graph, s, g)
    assert r.km == pytest.approx(shortest.km)
    assert r.cost == pytest.approx(c.calm_kg_per_km * r.km, rel=1e-4)
    assert r.cost == pytest.approx(c.path(r.lat, r.lon, r.on_link))


def test_moving_weather_is_taken_when_the_ship_gets_there():
    # Still at T0, a 20 m/s head wind (from the east) six hours later.
    fc = forecast()
    fc.fields["wind_u"][1] = -20.0
    frozen, moving = cost(fc), FuelCost(KCS_SHIP, fc, T0, 16 * KNOT, WAVES, frozen=False)
    leg = (0.0, 150.0, 0.0, 150.01)  # about 1 km: its midpoint is reached within a minute
    assert moving.kg(*leg, hours=0.0) == pytest.approx(frozen.kg(*leg), rel=1e-3)
    assert moving.kg(*leg, hours=6.0) > 1.1 * frozen.kg(*leg, hours=6.0)
    # Beyond the forecast's end the last step holds.
    assert moving.kg(*leg, hours=100.0) == pytest.approx(moving.kg(*leg, hours=6.0), rel=1e-3)
    assert moving.beyond_forecast == 2 + 1  # setting out at 6 h, the midpoint is just past the end too


def test_moving_search_in_steady_weather_matches_the_frozen_one(graph):
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    s, g = graph.nearest(0.5, 150.5), graph.nearest(5.5, 160.5)
    frozen, moving = cost(fc), FuelCost(KCS_SHIP, fc, T0, 16 * KNOT, WAVES, frozen=False)
    a = astar(graph, s, g, frozen.costs(graph), frozen.link_cost, lambda n: 0.0)
    b = astar(graph, s, g, moving.costs(graph), moving.link_cost, lambda n: 0.0)
    assert b.cost == pytest.approx(a.cost)
    assert b.cost == pytest.approx(moving.path(b.lat, b.lon, b.on_link))


def test_kernel_sample_matches_the_forecast():
    from shiprouting.route.kernel import sample

    rng = np.random.default_rng(0)
    fc = forecast()
    field = rng.normal(size=fc.fields["Hs"].shape)
    field[:, 100:103, 10:13] = np.nan  # a patch of land: (10.9, 11.4) is inside it, NaN both ways
    fc.fields["Hs"] = field
    seconds = fc.time.astype(np.float64)
    for lat, lon, h in [(10.3, 150.7, 2.5), (10.9, 11.4, 1.0), (-35.2, 179.8, 6.0), (-35.2, -179.6, 0.0), (89.9, 0.1, 3.0)]:
        t = T0 + np.timedelta64(int(h * 3600), "s")
        expected = fc.sample("Hs", lat, lon, t)
        got = sample(field, fc.lat, fc.lon, seconds, lat, lon, float(t.astype(np.float64)))
        assert got == pytest.approx(float(expected), nan_ok=True)
