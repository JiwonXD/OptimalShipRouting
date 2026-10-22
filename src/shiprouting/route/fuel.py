"""First attempt at fuel-optimal routing: A* on the sea graph with fuel as the cost.

The ship sails at one fixed speed through the water. A move costs the fuel
burnt over it, from the sea state at the move's midpoint in the forecast at the
departure time: the weather is frozen for the whole voyage, and currents are
left out. Links (canals, straits) are taken at the calm-water rate. The search
has no heuristic (Dijkstra): a following wind makes some moves cheaper than
calm water, so calm-water fuel for the distance left is not a lower bound.

With the weather frozen, every move's fuel is fixed before the search starts,
so it is computed for all moves at once, on arrays. At a fixed speed the fuel
rate depends only on the total resistance, so it is read from a table of the
ship model over resistance; wave added resistance comes from a table too.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from shiprouting.fuel.powering import brake_power
from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.fuel.wind import air_resistance
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import Graph, along
from shiprouting.weather.conditions import T01_PER_TM10
from shiprouting.weather.forecast import Forecast

KNOT = 1852 / 3600  # m/s


def bearing(lat_a, lon_a, lat_b, lon_b):
    """Initial great-circle course from a to b, rad clockwise from north."""
    p1, p2, dl = np.radians(lat_a), np.radians(lat_b), np.radians(np.subtract(lon_b, lon_a))
    return np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


@dataclass
class FuelCost:
    ship: Ship
    forecast: Forecast
    time: np.datetime64  # the weather used everywhere
    speed: float  # through the water, m/s
    waves: AddedResistanceTable | None = field(default=None, repr=False)  # built on first use if not given
    no_waves: int = 0  # moves with no wave data at the midpoint (taken without waves)
    over_mcr: int = 0  # moves that need more than the engine's MCR at this speed (fuel: inf)
    seconds: float = 0.0  # spent computing fuel
    _rate: tuple | None = field(default=None, repr=False)  # (resistance N, fuel kg/h) up to MCR

    @property
    def calm_kg_per_km(self) -> float:
        return self.ship.fuel_rate(self.speed) / (self.speed * 3.6)

    def fuel_rate(self, R):
        """Fuel, kg/h, at our speed against total resistance R (N); NaN beyond the engine's MCR."""
        if self._rate is None:
            ship = self.ship

            def power(r):
                return brake_power(r, self.speed, ship.factors, ship.propeller, ship.propeller_D)

            lo, hi = 0.2 * ship.resistance(self.speed), ship.resistance(self.speed)
            while power(hi) < ship.engine.mcr:
                hi *= 2
            for _ in range(50):  # bisect for the resistance at MCR
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if power(mid) < ship.engine.mcr else (lo, mid)
            R_grid = np.linspace(0.2 * ship.resistance(self.speed), lo, 500)
            self._rate = R_grid, np.array([ship.engine.fuel_rate(power(r)) for r in R_grid])
        R_grid, kg_h = self._rate
        return np.where(R <= R_grid[-1], np.interp(R, R_grid, kg_h), np.nan)

    def kg(self, lat_a, lon_a, lat_b, lon_b):
        """Fuel, kg, to sail each great circle from a to b; works on arrays.

        The sea state is that of weather_at and sea_state, on arrays.
        """
        t0 = time.perf_counter()
        if self.waves is None:
            self.waves = AddedResistanceTable.build(self.ship.seakeeping, self.speed)
        km = haversine_km(lat_a, lon_a, lat_b, lon_b)
        lat, lon = along(lat_a, lon_a, lat_b, lon_b, 0.5)
        heading = bearing(lat_a, lon_a, lat_b, lon_b)

        def get(name):
            return self.forecast.sample(name, lat, lon, self.time)

        u, v = get("wind_u"), get("wind_v")
        R = self.ship.resistance(self.speed) + air_resistance(
            self.ship.windage, np.hypot(u, v), np.arctan2(-u, -v) - heading, self.speed
        )
        Hs = get("Hs")
        no_waves = np.isnan(Hs)
        wave_from = np.arctan2(get("wave_from_x"), get("wave_from_y"))
        R = R + self.waves(np.where(no_waves, 0.0, Hs), T01_PER_TM10 * get("T_m10"), wave_from - heading)
        rate = self.fuel_rate(R)
        self.no_waves += int(no_waves.sum())
        self.over_mcr += int(np.isnan(rate).sum())
        self.seconds += time.perf_counter() - t0
        return np.where(np.isnan(rate), np.inf, rate * km / (self.speed * 3.6))

    def edge_costs(self, graph: Graph, chunk: int = 1_000_000) -> np.ndarray:
        """Fuel, kg, of every move of the graph, aligned with graph.indices, for quadtree.astar."""
        start = np.repeat(np.arange(len(graph.lat)), np.diff(graph.indptr))
        out = np.empty(len(graph.indices))
        for k in range(0, len(out), chunk):
            a, b = start[k : k + chunk], graph.indices[k : k + chunk]
            out[k : k + chunk] = self.kg(graph.lat[a], graph.lon[a], graph.lat[b], graph.lon[b])
        return out

    def link_cost(self, link, km: float) -> float:
        return km * self.calm_kg_per_km

    def path(self, lat, lon, on_link) -> float:
        """Fuel, kg, along a path, links at the calm-water rate."""
        sea = ~on_link
        a, b = np.nonzero(sea)[0], np.nonzero(sea)[0] + 1
        link_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])[on_link].sum()
        return float(self.kg(lat[a], lon[a], lat[b], lon[b]).sum() + link_km * self.calm_kg_per_km)


def run(graph: Graph, ship: Ship, forecast: Forecast, speed_kn: float, max_seconds: float) -> list[dict]:
    from shiprouting.sea.latlon import ROUTES
    from shiprouting.sea.quadtree import astar

    departure = forecast.time[0]
    t0 = time.perf_counter()
    cost = FuelCost(ship, forecast, departure, speed_kn * KNOT, AddedResistanceTable.build(ship.seakeeping, speed_kn * KNOT))
    print(f"wave added resistance table built in {time.perf_counter() - t0:.0f} s", flush=True)
    edge_kg = cost.edge_costs(graph)
    print(
        f"fuel of {len(edge_kg)} moves computed in {cost.seconds:.0f} s"
        f" (no wave data: {cost.no_waves}, beyond MCR: {cost.over_mcr})",
        flush=True,
    )
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "speed_kn": speed_kn, "weather_time": str(departure)}
        shortest = astar(graph, s, g)
        before = cost.no_waves
        row.update(
            shortest_km=round(shortest.km),
            shortest_fuel_t=round(cost.path(shortest.lat, shortest.lon, shortest.on_link) / 1000, 2),
            shortest_no_waves=f"{cost.no_waves - before}/{int((~shortest.on_link).sum())}",
            shortest_calm_fuel_t=round(shortest.km * cost.calm_kg_per_km / 1000, 2),
            shortest_seconds=round(shortest.seconds, 1),
        )
        try:
            r = astar(graph, s, g, edge_kg, cost.link_cost, lambda n: 0.0, max_seconds)
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
            before = cost.no_waves
            cost.path(r.lat, r.lon, r.on_link)
            row["fuel_no_waves"] = f"{cost.no_waves - before}/{int((~r.on_link).sum())}"
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
