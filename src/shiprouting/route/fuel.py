"""Fuel-optimal routing: a shortest-path search on the sea graph with fuel as the cost.

The ship sails at one fixed speed through the water. A move costs the fuel
burnt over it, from the sea state at the move's midpoint, and currents are left
out. The weather is either frozen at the departure time for the whole voyage,
or taken at the time the ship reaches the move's midpoint (moving weather),
held at the forecast's last step beyond its end. Links (canals, straits) are taken at the calm-water rate. The search
has no heuristic (Dijkstra): a following wind makes some moves cheaper than
calm water, so calm-water fuel for the distance left is not a lower bound.

In moving weather the search reaches a cell at the time of the cheapest path
to it, and the cell's moves are costed at that time as it is expanded. With the
weather frozen, every move's fuel is fixed before the search starts,
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
    frozen: bool = True  # weather at `time` everywhere; else at the time the ship gets there
    no_waves: int = 0  # moves with no wave data at the midpoint (taken without waves)
    over_mcr: int = 0  # moves that need more than the engine's MCR at this speed (fuel: inf)
    beyond_forecast: int = 0  # moves reached after the forecast's end (weather of its last step)
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

    def kg(self, lat_a, lon_a, lat_b, lon_b, hours=0.0):
        """Fuel, kg, to sail each great circle from a to b, setting out ``hours`` after `time`; works on arrays.

        The sea state is that of weather_at and sea_state, on arrays.
        """
        t0 = time.perf_counter()
        if self.waves is None:
            self.waves = AddedResistanceTable.build(self.ship.seakeeping, self.speed)
        lat_a, lon_a, lat_b, lon_b, hours = np.broadcast_arrays(lat_a, lon_a, lat_b, lon_b, hours)
        km = haversine_km(lat_a, lon_a, lat_b, lon_b)
        lat, lon = along(lat_a, lon_a, lat_b, lon_b, 0.5)
        heading = bearing(lat_a, lon_a, lat_b, lon_b)
        when = self.time
        if not self.frozen:
            seconds = np.round((hours + km / 2 / (self.speed * 3.6)) * 3600).astype("timedelta64[s]")
            when = self.time + seconds
            end = self.forecast.time[-1]
            self.beyond_forecast += int((when > end).sum())
            when = np.minimum(when, end)

        def get(name):
            return self.forecast.sample(name, lat, lon, when)

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

    def moving_costs(self, graph: Graph):
        """edge_cost function for quadtree.astar: the fuel of node c's moves, setting out km into the voyage."""

        def costs(c, km):
            lo, hi = graph.indptr[c], graph.indptr[c + 1]
            n = graph.indices[lo:hi]
            return self.kg(graph.lat[c], graph.lon[c], graph.lat[n], graph.lon[n], km / (self.speed * 3.6))

        return costs

    def link_cost(self, link, km: float) -> float:
        return km * self.calm_kg_per_km

    def path(self, lat, lon, on_link) -> float:
        """Fuel, kg, along a path, links at the calm-water rate."""
        seg_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        hours = (np.cumsum(seg_km) - seg_km) / (self.speed * 3.6)  # setting out on each segment
        a = np.nonzero(~on_link)[0]
        kg = self.kg(lat[a], lon[a], lat[a + 1], lon[a + 1], hours[a])
        return float(kg.sum() + seg_km[on_link].sum() * self.calm_kg_per_km)


def run(graph: Graph, ship: Ship, forecast: Forecast, speed_kn: float, max_seconds: float) -> list[dict]:
    """The shortest route, the fuel-optimal route in frozen weather and the one in moving weather,
    each costed in both kinds of weather."""
    from shiprouting.sea.latlon import ROUTES
    from shiprouting.sea.quadtree import astar

    departure = forecast.time[0]
    t0 = time.perf_counter()
    waves = AddedResistanceTable.build(ship.seakeeping, speed_kn * KNOT)
    print(f"wave added resistance table built in {time.perf_counter() - t0:.0f} s", flush=True)
    frozen = FuelCost(ship, forecast, departure, speed_kn * KNOT, waves)
    moving = FuelCost(ship, forecast, departure, speed_kn * KNOT, waves, frozen=False)
    edge_kg = frozen.edge_costs(graph)
    print(
        f"fuel of {len(edge_kg)} moves computed in {frozen.seconds:.0f} s"
        f" (no wave data: {frozen.no_waves}, beyond MCR: {frozen.over_mcr})",
        flush=True,
    )
    forecast_hours = float((forecast.time[-1] - departure) / np.timedelta64(1, "h"))
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "speed_kn": speed_kn, "departure": str(departure), "forecast_hours": forecast_hours}
        found = {"shortest": astar(graph, s, g), "frozen": astar(graph, s, g, edge_kg, frozen.link_cost, lambda n: 0.0)}
        try:
            found["moving"] = astar(graph, s, g, moving.moving_costs(graph), moving.link_cost, lambda n: 0.0, max_seconds)
        except TimeoutError as e:
            row["moving_error"] = str(e)
        for key, r in found.items():
            before = moving.beyond_forecast
            fuel_moving = moving.path(r.lat, r.lon, r.on_link)
            row[key] = {
                "km": round(r.km),
                "hours": round(r.km / (speed_kn * 1.852)),
                "fuel_frozen_t": round(frozen.path(r.lat, r.lon, r.on_link) / 1000, 2),
                "fuel_moving_t": round(fuel_moving / 1000, 2),
                "moves_after_forecast": f"{moving.beyond_forecast - before}/{int((~r.on_link).sum())}",
                "expanded": r.expanded,
                "seconds": round(r.seconds, 1),
                "links": r.links,
            }
        print(row, flush=True)
        for key, r in found.items():
            row[key] |= {"lat": np.round(r.lat, 4).tolist(), "lon": np.round(r.lon, 4).tolist()}
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Fuel-optimal routes for the test voyages, in frozen and in moving weather.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--speed-kn", type=float, default=16.0)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a moving-weather search after this long")
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
