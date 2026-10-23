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
so it is computed for all moves at once. Moves are costed one at a time by a
Numba kernel (kernel.fuel_kg), which is quick on arrays of any size. At a fixed speed the fuel
rate depends only on the total resistance, so it is read from a table of the
ship model over resistance; wave added resistance comes from a table too.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from shiprouting.fuel.powering import brake_power
from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.fuel.wind import RHO_AIR
from shiprouting.route import kernel
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import Graph
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
    _args: tuple | None = field(default=None, repr=False)  # for kernel.fuel_kg, this speed alone

    @property
    def calm_kg_per_km(self) -> float:
        return self.ship.fuel_rate(self.speed) / (self.speed * 3.6)

    def _tables(self) -> tuple:
        """Everything the kernel needs besides the moves, built on first use."""
        if self._args is None:
            ship = self.ship
            if self.waves is None:
                self.waves = AddedResistanceTable.build(ship.seakeeping, self.speed)

            def power(r):
                return brake_power(r, self.speed, ship.factors, ship.propeller, ship.propeller_D)

            # Fuel rate over total resistance: at a fixed speed it depends on nothing else.
            lo, hi = 0.2 * ship.resistance(self.speed), ship.resistance(self.speed)
            while power(hi) < ship.engine.mcr:
                hi *= 2
            for _ in range(50):  # bisect for the resistance at MCR
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if power(mid) < ship.engine.mcr else (lo, mid)
            R_grid = np.linspace(0.2 * ship.resistance(self.speed), lo, 500)
            kg_h = np.array([ship.engine.fuel_rate(power(r)) for r in R_grid])

            windage, fc = ship.windage, self.forecast
            seconds = fc.time.astype("datetime64[s]").astype(np.float64)
            self._args = (
                float(np.datetime64(self.time, "s").astype(np.float64)), self.frozen, np.array([self.speed]),
                np.array([ship.resistance(self.speed)], float),
                np.array(windage.angles if windage else (0.0, 180.0), float),
                np.array(windage.C_DA if windage else (0.0, 0.0), float),
                float(windage.A_XV if windage else 0.0), RHO_AIR,
                self.waves.T01, self.waves.angle, self.waves.per_Hs2[None],
                R_grid[None], kg_h[None],
                fc.lat, fc.lon, seconds,
                *(fc.fields[n] for n in ("wind_u", "wind_v", "Hs", "wave_from_x", "wave_from_y", "T_m10")),
            )  # fmt: skip
        return self._args

    def kg(self, lat_a, lon_a, lat_b, lon_b, hours=0.0):
        """Fuel, kg, to sail each great circle from a to b, setting out ``hours`` after `time`; works on arrays.

        The sea state is that of weather_at and sea_state. Moves that need more than the
        engine's MCR cost inf.
        """
        return fuel_kg([self], self._tables(), lat_a, lon_a, lat_b, lon_b, hours)[0]

    def edge_costs(self, graph: Graph, chunk: int = 1_000_000) -> np.ndarray:
        """Fuel, kg, of every move of the graph, aligned with graph.indices, for quadtree.astar."""
        start = np.repeat(np.arange(len(graph.lat)), np.diff(graph.indptr))
        out = np.empty(len(graph.indices))
        for k in range(0, len(out), chunk):
            a, b = start[k : k + chunk], graph.indices[k : k + chunk]
            out[k : k + chunk] = self.kg(graph.lat[a], graph.lon[a], graph.lat[b], graph.lon[b])
        return out

    def costs(self, graph: Graph):
        """edge_cost function for quadtree.astar, its clock in hours since `time`: the fuel of node
        c's moves and their hours. In frozen weather all moves are costed up front."""
        edge_kg = self.edge_costs(graph) if self.frozen else None
        per_hour = self.speed * 3.6  # km

        def costs(c, hours):
            lo, hi = graph.indptr[c], graph.indptr[c + 1]
            if edge_kg is not None:
                return edge_kg[lo:hi], graph.km[lo:hi] / per_hour
            n = graph.indices[lo:hi]
            return self.kg(graph.lat[c], graph.lon[c], graph.lat[n], graph.lon[n], hours), graph.km[lo:hi] / per_hour

        return costs

    def link_cost(self, link, km: float, hours: float) -> tuple[float, float]:
        """link_cost for quadtree.astar: links at the calm-water rate."""
        return km * self.calm_kg_per_km, km / (self.speed * 3.6)

    def path(self, lat, lon, on_link) -> float:
        """Fuel, kg, along a path, links at the calm-water rate."""
        seg_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        hours = (np.cumsum(seg_km) - seg_km) / (self.speed * 3.6)  # setting out on each segment
        a = np.nonzero(~on_link)[0]
        kg = self.kg(lat[a], lon[a], lat[a + 1], lon[a + 1], hours[a])
        return float(kg.sum() + seg_km[on_link].sum() * self.calm_kg_per_km)


PER_SPEED = (2, 3, 10, 11, 12)  # in FuelCost._tables: speed, R_hull, per_Hs2, R_grid, kg_h


def stacked_tables(costs: list[FuelCost]) -> tuple:
    """The kernel's arguments for several speeds at once. They must share the forecast, the
    departure, frozen, the windage and the wave table's grids."""
    args = [c._tables() for c in costs]
    return tuple(np.concatenate([a[i] for a in args]) if i in PER_SPEED else args[0][i] for i in range(len(args[0])))


def fuel_kg(costs: list[FuelCost], args: tuple, lat_a, lon_a, lat_b, lon_b, hours=0.0) -> np.ndarray:
    """FuelCost.kg at each of costs' speeds, shape (speeds, *moves); args from stacked_tables."""
    t0 = time.perf_counter()
    moves = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (lat_a, lon_a, lat_b, lon_b, hours)))
    shape = moves[0].shape
    out, counts = np.empty((len(costs), moves[0].size)), np.zeros((len(costs), 3), np.int64)
    kernel.fuel_kg(*(np.ascontiguousarray(v).ravel() for v in moves), out, counts, *args)
    seconds = (time.perf_counter() - t0) / len(costs)
    for c, n in zip(costs, counts):
        c.no_waves += int(n[0])
        c.over_mcr += int(n[1])
        c.beyond_forecast += int(n[2])
        c.seconds += seconds
    return out.reshape(len(costs), *shape)

def run(graph: Graph, ship: Ship, forecast: Forecast, speed_kn: float, max_seconds: float, land=None) -> list[dict]:
    """The shortest route, the fuel-optimal route in frozen weather and the one in moving weather,
    each costed in both kinds of weather. Given the land, each also with its sharp turns smoothed out."""
    from shiprouting.sea.latlon import ROUTES, turn_deg
    from shiprouting.sea.quadtree import astar, smooth_turns

    departure = forecast.time[0]
    t0 = time.perf_counter()
    waves = AddedResistanceTable.build(ship.seakeeping, speed_kn * KNOT)
    print(f"wave added resistance table built in {time.perf_counter() - t0:.0f} s", flush=True)
    frozen = FuelCost(ship, forecast, departure, speed_kn * KNOT, waves)
    moving = FuelCost(ship, forecast, departure, speed_kn * KNOT, waves, frozen=False)
    frozen_costs = frozen.costs(graph)
    print(
        f"fuel of {len(graph.indices)} moves computed in {frozen.seconds:.0f} s"
        f" (no wave data: {frozen.no_waves}, beyond MCR: {frozen.over_mcr})",
        flush=True,
    )
    forecast_hours = float((forecast.time[-1] - departure) / np.timedelta64(1, "h"))
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "speed_kn": speed_kn, "departure": str(departure), "forecast_hours": forecast_hours}
        found = {"shortest": astar(graph, s, g), "frozen": astar(graph, s, g, frozen_costs, frozen.link_cost, lambda n: 0.0)}
        try:
            found["moving"] = astar(graph, s, g, moving.costs(graph), moving.link_cost, lambda n: 0.0, max_seconds)
        except TimeoutError as e:
            row["moving_error"] = str(e)
        if land is not None:
            found |= {f"{key}_smoothed": smooth_turns(land, r) for key, r in found.items()}
        for key, r in found.items():
            before = moving.beyond_forecast
            fuel_moving = moving.path(r.lat, r.lon, r.on_link)
            row[key] = {
                "km": round(r.km),
                "hours": round(r.km / (speed_kn * 1.852)),
                "fuel_frozen_t": round(frozen.path(r.lat, r.lon, r.on_link) / 1000, 2),
                "fuel_moving_t": round(fuel_moving / 1000, 2),
                "moves_after_forecast": f"{moving.beyond_forecast - before}/{int((~r.on_link).sum())}",
                "turns_over_60": int((turn_deg(r.lat, r.lon) >= 60).sum()),
                "expanded": r.expanded,
                "seconds": round(r.seconds, 1),
                "links": r.links,
            }
        print(row, flush=True)
        for key, r in found.items():
            row[key] |= {"lat": np.round(r.lat, 4).tolist(), "lon": np.round(r.lon, 4).tolist(), "on_link": r.on_link.tolist()}
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
    parser.add_argument("--bathymetry", help="GEBCO grid, netCDF: keep out of water --min-depth m deep or less")
    parser.add_argument("--min-depth", type=float, default=15.0)
    parser.add_argument("--depth-check", choices=["centre", "lines"], default="lines",
                        help="check only cell centres, or also every move and smoothed leg")
    args = parser.parse_args()

    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    t0 = time.perf_counter()
    bathymetry = None
    if args.bathymetry:
        from shiprouting.sea.depth import Bathymetry, LandOrShallow

        bathymetry = Bathymetry(args.bathymetry)
        if args.depth_check == "lines":
            land, bathymetry = LandOrShallow(land, bathymetry, args.min_depth), None
    graph = Graph.build(land, min_step=args.min_step, links=links, bathymetry=bathymetry, min_depth=args.min_depth)
    print(f"graph: {len(graph.lat)} cells, {len(graph.links)} links, built in {time.perf_counter() - t0:.0f} s", flush=True)
    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    rows = run(graph, KCS_SHIP, forecast, args.speed_kn, args.max_seconds, land)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
