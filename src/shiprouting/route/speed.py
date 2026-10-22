"""First attempt at choosing the speed: each move at whichever of a few speeds burns the least fuel.

Every speed has its own FuelCost in moving weather. When the search expands a cell, each
move is costed at every speed, at the time the ship reaches the cell, and the cheapest is
taken; the clock (hours) advances by that move's time at that speed. Links are taken at
the cheapest calm-water speed. There is no arrival time to meet.
"""

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph, link_jumps, trace


@dataclass
class SpeedChoice:
    costs: list[FuelCost]  # one per speed, moving weather, same departure

    def edge_cost(self, graph: Graph):
        """edge_cost function for quadtree.astar, its clock in hours."""
        per_speed = [c.costs(graph) for c in self.costs]

        def costs(c, hours):
            kg, h = (np.array(v) for v in zip(*(f(c, hours) for f in per_speed)))  # (speeds, moves)
            pick = np.argmin(kg, axis=0)
            moves = np.arange(kg.shape[1])
            return kg[pick, moves], h[pick, moves]

        return costs

    def link_cost(self, link, km: float, hours: float) -> tuple[float, float]:
        return min((c.link_cost(link, km, hours) for c in self.costs), key=lambda kg_h: kg_h[0])

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


def eta_search(graph: Graph, s: int, g: int, choice: SpeedChoice, eta_hours: float, bin_hours: float = 1.0,
               max_seconds: float = math.inf, max_states: float = math.inf) -> SearchResult:
    """Least fuel from s to g arriving within eta_hours, each move at any of the choice's speeds.

    The time is part of the search state: a cell reached in different hours (bins of
    bin_hours) is a different state, because the weather ahead depends on when the ship
    gets there. Moves after which even the top speed could not reach g in time (great
    circle) are dropped.
    """
    t0 = time.perf_counter()
    per_speed = [c.costs(graph) for c in choice.costs]
    top_kmh = max(c.speed for c in choice.costs) * 3.6

    def too_late(n, hours):
        return hours + float(haversine_km(graph.lat[n], graph.lon[n], graph.lat[g], graph.lon[g])) / top_kmh > eta_hours

    jumps = link_jumps(graph)

    start = (s, 0)
    best, clock, parent = {start: 0.0}, {start: 0.0}, {start: None}
    closed, open_list, expanded = set(), [(0.0, start)], 0
    while open_list:
        _, state = heapq.heappop(open_list)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1
        c = state[0]
        if c == g:
            break
        if time.perf_counter() - t0 > max_seconds or len(best) > max_states:
            raise TimeoutError(f"stopped after {time.perf_counter() - t0:.0f} s ({expanded} states expanded, {len(best)} reached)")
        lo, hi = graph.indptr[c], graph.indptr[c + 1]
        moves = []
        for f in per_speed:
            kg, h = f(c, clock[state])
            moves += [(int(n), float(w), float(dt), None, False) for n, w, dt in zip(graph.indices[lo:hi], kg, h)]
        for n, km, link, rev in jumps.get(c, []):
            moves += [(n, *cost.link_cost(link, km, clock[state]), link, rev) for cost in choice.costs]
        for n, w, dt, link, rev in moves:
            hours = clock[state] + dt
            if too_late(n, hours):
                continue
            nxt = (n, int(hours // bin_hours))
            cost = best[state] + w
            if nxt not in closed and cost < best.get(nxt, math.inf):
                best[nxt], clock[nxt], parent[nxt] = cost, hours, (state, link, rev)
                heapq.heappush(open_list, (cost, nxt))
    else:
        raise ValueError(f"no route arrives within {eta_hours:g} h ({expanded} states expanded)")

    steps = [(state, None, False)]  # from the goal back
    while parent[steps[-1][0]] is not None:
        prev, link, rev = parent[steps[-1][0]]
        steps[-1] = (steps[-1][0], link, rev)
        steps.append((prev, None, False))
    steps = [(st[0], clock[st], link, rev) for st, link, rev in reversed(steps)]
    return trace(graph, steps, expanded, time.perf_counter() - t0, best[state])


def run(graph: Graph, ship, forecast, speeds_kn, fixed_kn: float, bin_hours: float, max_seconds: float, max_states: float) -> list[dict]:
    """For each test voyage, the moving-weather route at a fixed speed against the least-fuel one
    choosing speeds that arrives no later."""
    from shiprouting.fuel.waves import AddedResistanceTable
    from shiprouting.sea.latlon import ROUTES
    from shiprouting.sea.quadtree import astar

    departure = forecast.time[0]
    costs = {}
    for kn in sorted({*speeds_kn, fixed_kn}):
        t0 = time.perf_counter()
        waves = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        costs[kn] = FuelCost(ship, forecast, departure, kn * KNOT, waves, frozen=False)
        print(f"{kn:g} kn: wave added resistance table built in {time.perf_counter() - t0:.0f} s", flush=True)
    fixed, choice = costs[fixed_kn], SpeedChoice([costs[kn] for kn in speeds_kn])
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "departure": str(departure)}
        found = {"fixed": astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)}
        eta = float(found["fixed"].clock[-1])
        try:
            found["eta"] = eta_search(graph, s, g, choice, eta, bin_hours, max_seconds, max_states)
        except (TimeoutError, ValueError) as e:
            row["eta"] = {"error": str(e)}
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
    parser.add_argument("--bin-hours", type=float, default=1.0, help="time resolution of the search state")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    parser.add_argument("--max-states", type=float, default=2e7, help="give up a search after reaching this many states")
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
    rows = run(graph, KCS_SHIP, forecast, speeds, args.fixed_kn, args.bin_hours, args.max_seconds, args.max_states)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
