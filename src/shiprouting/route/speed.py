"""Route and speed together: the least fuel arriving by a given time, each move at one of a few speeds.

Every speed has its own FuelCost in moving weather. When the search expands a cell, each
move is costed at every speed, at the time the ship reaches the cell; the clock (hours)
advances by that move's time at the speed taken. Links are taken at calm-water rates.

Time is not part of the search state. Instead each hour is given a price in kg of fuel
(per_hour), each move takes the speed with the least fuel + price × hours, and the
search minimises that sum. Price 0 sails at the slowest speed; a higher price sails
faster. meet_eta bisects the price until the route arrives just within the given time.
"""

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np

from shiprouting.route.fuel import KNOT, FuelCost, fuel_kg, stacked_tables
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph, astar


@dataclass
class SpeedChoice:
    costs: list[FuelCost]  # one per speed, slowest first, moving weather, same departure
    per_hour: float = 0.0  # the price of an hour, kg of fuel
    _args: tuple | None = field(default=None, repr=False)  # for kernel.fuel_kg, all speeds

    def kg(self, lat_a, lon_a, lat_b, lon_b, hours=0.0) -> np.ndarray:
        """FuelCost.kg at every speed in one pass, shape (speeds, *moves)."""
        if self._args is None:
            self._args = stacked_tables(self.costs)
        return fuel_kg(self.costs, self._args, lat_a, lon_a, lat_b, lon_b, hours)

    def edge_cost(self, graph: Graph):
        """edge_cost function for quadtree.astar, its clock in hours: fuel + per_hour × hours."""
        km_h = np.array([c.speed * 3.6 for c in self.costs])[:, None]

        def costs(c, hours):
            lo, hi = graph.indptr[c], graph.indptr[c + 1]
            n = graph.indices[lo:hi]
            kg = self.kg(graph.lat[c], graph.lon[c], graph.lat[n], graph.lon[n], hours)  # (speeds, moves)
            h = graph.km[lo:hi] / km_h
            total = kg + self.per_hour * h
            pick = np.argmin(total, axis=0)
            moves = np.arange(kg.shape[1])
            return total[pick, moves], h[pick, moves]

        return costs

    def link_cost(self, link, km: float, hours: float) -> tuple[float, float]:
        kg, h = min((c.link_cost(link, km, hours) for c in self.costs), key=lambda kg_h: kg_h[0] + self.per_hour * kg_h[1])
        return kg + self.per_hour * h, h

    def speeds_kn(self, r: SearchResult) -> np.ndarray:
        """The speed of each segment of a path found with edge_cost, knots."""
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        return seg_km / np.diff(r.clock) / 1.852

    def path(self, r: SearchResult) -> float:
        """Fuel, kg, along a path found with edge_cost, each segment at its own speed and time."""
        kn = self.speeds_kn(r)
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        kg = 0.0
        for c in self.costs:
            mine = np.isclose(kn, c.speed / KNOT, rtol=1e-6)
            kg += seg_km[mine & r.on_link].sum() * c.calm_kg_per_km
            a = np.nonzero(mine & ~r.on_link)[0]
            kg += c.kg(r.lat[a], r.lon[a], r.lat[a + 1], r.lon[a + 1], r.clock[a]).sum()
        return float(kg)

    def fill_slack(self, r: SearchResult, eta_hours: float, rounds: int = 20) -> SearchResult:
        """Spend the time a found path leaves before eta_hours: step segments down to the next
        slower speed, those saving the most fuel per extra hour first. Each round re-costs the
        whole path at its new times and stops when that no longer saves fuel."""
        speeds = np.array([c.speed for c in self.costs])  # m/s
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        seg = np.arange(len(seg_km))
        sea = np.nonzero(~r.on_link)[0]

        def evaluate(k):
            clock = np.r_[0.0, np.cumsum(seg_km / (speeds[k] * 3.6))]
            kg = seg_km * np.array([c.calm_kg_per_km for c in self.costs])[:, None]  # every speed, setting out at the profile's times
            kg[:, sea] = self.kg(r.lat[sea], r.lon[sea], r.lat[sea + 1], r.lon[sea + 1], clock[sea])
            return clock, kg, kg[k, seg].sum()

        k = np.argmin(np.abs(self.speeds_kn(r)[:, None] - speeds[None, :] / KNOT), axis=1)
        clock, kg, fuel = evaluate(k)
        for _ in range(rounds):
            slower = np.maximum(k - 1, 0)
            extra = seg_km / (speeds[slower] * 3.6) - seg_km / (speeds[k] * 3.6)
            saving = kg[k, seg] - kg[slower, seg]
            worth = np.nonzero((k > 0) & (saving > 0))[0]
            worth = worth[np.argsort(-saving[worth] / extra[worth])]
            take = worth[np.cumsum(extra[worth]) <= eta_hours - clock[-1]]
            if len(take) == 0:
                break
            k2 = k.copy()
            k2[take] -= 1
            clock2, kg2, fuel2 = evaluate(k2)
            if fuel2 >= fuel:
                break
            k, clock, kg, fuel = k2, clock2, kg2, fuel2
        return dataclasses.replace(r, clock=clock, cost=float(fuel))


def meet_eta(graph: Graph, s: int, g: int, choice: SpeedChoice, eta_hours: float, tolerance: float = 0.002,
             max_tries: int = 12, max_seconds: float = math.inf) -> tuple[SearchResult | None, list[dict]]:
    """The least-fuel route found arriving within eta_hours, bisecting the price of an hour.

    Stops when a route arrives within a fraction `tolerance` of the voyage before the ETA, or after max_tries
    searches. The best route that arrived in time then spends the time it has left
    (SpeedChoice.fill_slack). Returns that route (None if none arrived in time) and a log
    of the tries.
    """
    tries, best = [], None

    def search(per_hour):
        choice.per_hour = per_hour
        r = astar(graph, s, g, choice.edge_cost(graph), choice.link_cost, lambda n: 0.0, max_seconds)
        fuel = r.cost - per_hour * float(r.clock[-1])
        tries.append({"per_hour": round(per_hour, 1), "hours": round(float(r.clock[-1]), 1), "fuel_t": round(fuel / 1000, 2),
                      "seconds": round(r.seconds, 1)})
        print(f"  price {per_hour:8.1f} kg/h: {r.clock[-1]:7.1f} h, {fuel / 1000:8.2f} t, {r.seconds:5.1f} s", flush=True)
        return r, fuel

    def in_time(r):
        return r.clock[-1] <= eta_hours * (1 + 1e-9)  # a route sailed at exactly the speed for the ETA, give or take rounding

    lo, hi = 0.0, 1000.0
    while len(tries) < max_tries:
        r, fuel = search(hi)
        if in_time(r):
            best = (r, fuel)
            break
        lo, hi = hi, hi * 4
    if best is not None and lo == 0.0 and len(tries) < max_tries:  # in time at the first price: perhaps even at the slowest
        r, fuel = search(0.0)
        if in_time(r):
            best = (r, fuel)
            hi = 0.0
    while best is not None and len(tries) < max_tries and hi > 0.0 and best[0].clock[-1] < eta_hours * (1 - tolerance):
        mid = (lo + hi) / 2
        r, fuel = search(mid)
        if in_time(r):
            hi = mid
            if fuel < best[1]:
                best = (r, fuel)
        else:
            lo = mid
    if best is None:
        return None, tries
    r, fuel = best
    return choice.fill_slack(dataclasses.replace(r, cost=fuel), eta_hours), tries


def run(graph: Graph, ship, forecast, speeds_kn, fixed_kn: float, max_tries: int, max_seconds: float, tables: str | None = None) -> list[dict]:
    """For each test voyage, the moving-weather route at a fixed speed against the least-fuel one
    choosing speeds that arrives no later. Wave tables are kept in the folder `tables` if given."""
    from shiprouting.fuel.waves import AddedResistanceTable
    from shiprouting.sea.latlon import ROUTES

    departure = forecast.time[0]
    costs = {}
    for kn in sorted({*speeds_kn, fixed_kn}):
        t0 = time.perf_counter()
        if tables:
            waves = AddedResistanceTable.cached(tables, ship.seakeeping, kn * KNOT)
        else:
            waves = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        costs[kn] = FuelCost(ship, forecast, departure, kn * KNOT, waves, frozen=False)
        print(f"{kn:g} kn: wave added resistance table ready in {time.perf_counter() - t0:.0f} s", flush=True)
    fixed, choice = costs[fixed_kn], SpeedChoice([costs[kn] for kn in sorted(speeds_kn)])
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "departure": str(departure)}
        print(name, flush=True)
        found = {"fixed": astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)}
        eta = float(found["fixed"].clock[-1])
        t0 = time.perf_counter()
        r, row["tries"] = meet_eta(graph, s, g, choice, eta, max_tries=max_tries, max_seconds=max_seconds)
        row["eta_seconds"] = round(time.perf_counter() - t0, 1)
        if r is not None:
            found["eta"] = r
        for key, r in found.items():
            kn = choice.speeds_kn(r)
            row[key] = {
                "km": round(r.km),
                "hours": round(float(r.clock[-1]), 1),
                "fuel_t": round(r.cost / 1000, 2),
                "speeds_kn": {f"{v:g}": int(n) for v, n in zip(*np.unique(np.round(kn, 3), return_counts=True))},
                "expanded": r.expanded,
                "seconds": round(r.seconds, 1),
                "links": r.links,
                "lat": np.round(r.lat, 4).tolist(),
                "lon": np.round(r.lon, 4).tolist(),
                "clock": np.round(r.clock, 2).tolist(),
            }
        print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "clock")} if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Least-fuel routes and speeds arriving no later than a fixed speed, in moving weather.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--speeds-kn", default="12,14,16,18,20", help="speeds to choose from, comma-separated")
    parser.add_argument("--fixed-kn", type=float, default=16.0, help="the fixed speed to compare with")
    parser.add_argument("--max-tries", type=int, default=12, help="searches per voyage to meet the arrival time")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
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
    speeds = [float(v) for v in args.speeds_kn.split(",")]
    rows = run(graph, KCS_SHIP, forecast, speeds, args.fixed_kn, args.max_tries, args.max_seconds, args.tables)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
