"""First attempt at fuel-optimal routing: A* on the sea graph with fuel as the cost.

The ship sails at one fixed speed through the water. A move costs the fuel
burnt over it, from the sea state at the move's midpoint in the forecast at the
departure time: the weather is frozen for the whole voyage, and currents are
left out. Wave added resistance comes from a table built once for the speed. Links (canals, straits) are taken at the calm-water rate. The A*
heuristic is the calm-water fuel for the great-circle distance left.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from shiprouting.fuel.powering import brake_power
from shiprouting.fuel.ship import SeaState, Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import Graph, along
from shiprouting.weather.conditions import sea_state, weather_at
from shiprouting.weather.forecast import Forecast

KNOT = 1852 / 3600  # m/s


def bearing(lat_a, lon_a, lat_b, lon_b) -> float:
    """Initial great-circle course from a to b, rad clockwise from north."""
    p1, p2, dl = math.radians(lat_a), math.radians(lat_b), math.radians(lon_b - lon_a)
    return math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))


@dataclass
class FuelCost:
    ship: Ship
    forecast: Forecast
    time: np.datetime64  # the weather used everywhere
    speed: float  # through the water, m/s
    waves: AddedResistanceTable | None = field(default=None, repr=False)  # built on first use if not given
    evaluations: int = 0  # moves whose fuel was computed with the full sea state
    no_waves: int = 0  # of those, moves with no wave data at the midpoint (taken without waves)
    seconds: float = 0.0  # spent computing fuel
    _cache: dict = field(default_factory=dict, repr=False)

    def fuel_rate(self, sea: SeaState) -> float:
        """Like ``Ship.fuel_rate`` at our speed, kg/h, with wave added resistance from the table."""
        if sea.waves and self.waves is None:
            self.waves = AddedResistanceTable.build(self.ship.seakeeping, self.speed)
        R = self.ship.total_resistance(self.speed, SeaState(sea.wind_speed, sea.wind_angle))
        for w in sea.waves:
            assert w.s == 1.0, "the table is for wind-sea spreading"
            R += self.waves(w.Hs, w.T01, w.angle)
        ship = self.ship
        return ship.engine.fuel_rate(brake_power(R, self.speed, ship.factors, ship.propeller, ship.propeller_D))

    @property
    def calm_kg_per_km(self) -> float:
        return self.ship.fuel_rate(self.speed) / (self.speed * 3.6)

    def between(self, lat_a, lon_a, lat_b, lon_b) -> float:
        """Fuel, kg, to sail the great circle from a to b."""
        t0 = time.perf_counter()
        km = float(haversine_km(lat_a, lon_a, lat_b, lon_b))
        mid_lat, mid_lon = (float(v) for v in along(lat_a, lon_a, lat_b, lon_b, 0.5))
        heading = bearing(lat_a, lon_a, lat_b, lon_b)
        weather = weather_at(self.forecast, mid_lat, mid_lon, self.time)
        try:
            sea = sea_state(weather, heading)
        except ValueError:
            self.no_waves += 1
            sea = SeaState(math.hypot(weather.wind_u, weather.wind_v), math.atan2(-weather.wind_u, -weather.wind_v) - heading)
        kg = self.fuel_rate(sea) * km / (self.speed * 3.6)
        self.evaluations += 1
        self.seconds += time.perf_counter() - t0
        return kg

    def weight(self, graph: Graph):
        """Move cost for quadtree.astar."""

        def w(c, n, km, link) -> float:
            if link is not None:
                return km * self.calm_kg_per_km
            if (c, n) not in self._cache:
                self._cache[c, n] = self.between(graph.lat[c], graph.lon[c], graph.lat[n], graph.lon[n])
            return self._cache[c, n]

        return w

    def heuristic(self, graph: Graph, g: int):
        per_km = self.calm_kg_per_km
        return lambda n: per_km * float(haversine_km(graph.lat[n], graph.lon[n], graph.lat[g], graph.lon[g]))

    def path(self, lat, lon, on_link) -> float:
        """Fuel, kg, along a path, links at the calm-water rate."""
        kg = 0.0
        for k in range(len(lat) - 1):
            if on_link[k]:
                kg += float(haversine_km(lat[k], lon[k], lat[k + 1], lon[k + 1])) * self.calm_kg_per_km
            else:
                kg += self.between(lat[k], lon[k], lat[k + 1], lon[k + 1])
        return kg


def run(graph: Graph, ship: Ship, forecast: Forecast, speed_kn: float, max_seconds: float) -> list[dict]:
    from shiprouting.sea.latlon import ROUTES
    from shiprouting.sea.quadtree import astar

    departure = forecast.time[0]
    t0 = time.perf_counter()
    waves = AddedResistanceTable.build(ship.seakeeping, speed_kn * KNOT)
    print(f"wave added resistance table built in {time.perf_counter() - t0:.0f} s", flush=True)
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "speed_kn": speed_kn, "weather_time": str(departure)}
        shortest = astar(graph, s, g)
        check = FuelCost(ship, forecast, departure, speed_kn * KNOT, waves)
        row.update(
            shortest_km=round(shortest.km),
            shortest_fuel_t=round(check.path(shortest.lat, shortest.lon, shortest.on_link) / 1000, 2),
            shortest_calm_fuel_t=round(shortest.km * check.calm_kg_per_km / 1000, 2),
            shortest_seconds=round(shortest.seconds, 1),
        )
        cost = FuelCost(ship, forecast, departure, speed_kn * KNOT, waves)
        try:
            r = astar(graph, s, g, cost.weight(graph), cost.heuristic(graph, g), max_seconds)
        except (ValueError, TimeoutError) as e:
            row["error"] = str(e)
        else:
            row.update(
                fuel_km=round(r.km),
                fuel_t=round(r.cost / 1000, 2),
                expanded=r.expanded,
                seconds=round(r.seconds, 1),
                links=r.links,
                lat=np.round(r.lat, 4).tolist(),
                lon=np.round(r.lon, 4).tolist(),
            )
        row.update(evaluations=cost.evaluations, fuel_seconds=round(cost.seconds, 1), no_waves=cost.no_waves)
        print({k: v for k, v in row.items() if k not in ("lat", "lon")}, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Fuel-optimal routes for the test voyages, weather frozen at departure.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--speed-kn", type=float, default=16.0)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a fuel search after this long")
    args = parser.parse_args()

    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    t0 = time.perf_counter()
    graph = Graph.build(land, min_step=args.min_step, links=links)
    print(f"graph: {len(graph.lat)} cells, built in {time.perf_counter() - t0:.0f} s", flush=True)
    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    rows = run(graph, KCS_SHIP, forecast, args.speed_kn, args.max_seconds)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
