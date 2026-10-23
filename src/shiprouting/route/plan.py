"""Plan one voyage: leave at a given time, arrive by a deadline; the shortest route and the
least-fuel one, compared.

The shortest route sails at the one speed that arrives at the deadline (no slower than the
slowest speed offered; if even the fastest is too slow, the deadline cannot be met). The
least-fuel route picks its way and a speed per move from those offered, arriving by the
deadline (speed.meet_eta). Both are costed in the same forecast, in moving weather from the
departure, so the saving is what choosing the way and the speeds is worth.

Waypoints are passed in order, each leg searched after the last. The least-fuel route
either gives each leg a share of the time in proportion to its shortest distance
(legs="distance"), or searches all legs at one price of an hour and meets the deadline
for the voyage as a whole (legs="shared"). Given the land, both routes then have their
sharp turns smoothed out (quadtree.smooth_turns), waypoints kept, and the least-fuel
route is re-timed to the speeds on offer and spends the time that saves.
"""

import math
import time

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.speed import SpeedChoice, meet_eta
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph, astar, join_legs, smooth_turns
from shiprouting.weather.forecast import Forecast

SPEEDS_KN = (12, 14, 16, 18, 20, 22, 24)


def plan(graph: Graph, ship: Ship, forecast: Forecast, start, goal, departure, deadline, speeds_kn=SPEEDS_KN,
         tables: str | None = None, max_tries: int = 12, max_seconds: float = math.inf, via=(), legs: str = "distance",
         land=None) -> dict:
    departure, deadline = np.datetime64(departure, "s"), np.datetime64(deadline, "s")
    if not forecast.time[0] <= departure < forecast.time[-1]:
        raise ValueError(f"departure {departure} is outside the forecast")
    hours = float((deadline - departure) / np.timedelta64(1, "h"))
    points = [start, *via, goal]
    nodes = [graph.nearest(*p) for p in points]
    out = {
        "start": list(start), "goal": list(goal), "via": [list(p) for p in via], "legs_time": legs,
        "departure": str(departure), "deadline": str(deadline), "hours": hours,
        "moved_km": [round(float(haversine_km(graph.lat[n], graph.lon[n], *p)), 1) for n, p in zip(nodes, points)],
    }

    waves = {}

    def cost(kn, at=departure):
        if kn not in waves:
            if tables:
                waves[kn] = AddedResistanceTable.cached(tables, ship.seakeeping, kn * KNOT)
            else:
                waves[kn] = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        return FuelCost(ship, forecast, at, kn * KNOT, waves[kn], frozen=False)

    def waypoints(r):
        """Indices of the waypoints in path r."""
        at = [0]
        for n in nodes[1:-1]:
            here = np.isclose(r.lat[at[-1]:], graph.lat[n]) & np.isclose((r.lon[at[-1]:] - graph.lon[n] + 180) % 360 - 180, 0)
            at.append(at[-1] + int(np.argmax(here)))
        return at[1:]

    def smoothed(r):
        return r if land is None else smooth_turns(land, r, keep_points=waypoints(r))

    t0 = time.perf_counter()
    shortest = [astar(graph, a, b) for a, b in zip(nodes[:-1], nodes[1:])]
    r = smoothed(join_legs(shortest))
    need_kn = r.km / hours / 1.852
    if need_kn > max(speeds_kn):
        out["error"] = f"the deadline cannot be met: the shortest route ({r.km:.0f} km) needs {need_kn:.1f} kn"
        return out
    kn = max(need_kn, min(speeds_kn))
    out["shortest"] = describe(r, cost(kn).path(r.lat, r.lon, r.on_link), np.full(len(r.lat) - 1, kn), time.perf_counter() - t0)

    t0 = time.perf_counter()
    # The shortest route's own speed is offered too, so sailing it as it does is among the choices.
    offered = sorted({*speeds_kn, kn})
    choice = SpeedChoice([cost(k) for k in offered])
    if legs == "shared":
        f, out["tries"] = meet_eta(graph, nodes[0], nodes[-1], choice, hours, max_tries=max_tries, max_seconds=max_seconds,
                                   via=nodes[1:-1])
    else:
        found, out["tries"], at = [], [], departure
        for (a, b), leg in zip(zip(nodes[:-1], nodes[1:]), shortest):
            f, tries = meet_eta(graph, a, b, SpeedChoice([cost(k, at) for k in offered]), hours * leg.km / sum(r.km for r in shortest),
                                max_tries=max_tries, max_seconds=max_seconds)
            out["tries"].append(tries)
            if f is None:
                break
            found.append(f)
            at = at + np.timedelta64(int(round(f.clock[-1] * 3600)), "s")
        f = join_legs(found) if f is not None else None
    if f is None:
        out["error"] = "no route choosing speeds arrived by the deadline"
        return out
    if land is not None:
        f = choice.fill_slack(choice.snap_up(smoothed(f)), hours)
    out["least_fuel"] = describe(f, f.cost, choice.speeds_kn(f), time.perf_counter() - t0)
    ends = [0, *waypoints(f), len(f.lat) - 1]
    seg_km = np.r_[0.0, np.cumsum(haversine_km(f.lat[:-1], f.lon[:-1], f.lat[1:], f.lon[1:]))]
    out["least_fuel"]["legs"] = [{"km": round(float(seg_km[b] - seg_km[a]), 1), "hours": round(float(f.clock[b] - f.clock[a]), 1)}
                                 for a, b in zip(ends[:-1], ends[1:])]
    saving = out["shortest"]["fuel_t"] - out["least_fuel"]["fuel_t"]
    out["saving_t"] = round(saving, 2)
    out["saving_pct"] = round(100 * saving / out["shortest"]["fuel_t"], 2)
    return out


def describe(r: SearchResult, fuel_kg: float, speeds_kn, seconds: float) -> dict:
    seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
    return {
        "km": round(r.km, 1),
        "hours": round(float((seg_km / (np.asarray(speeds_kn) * 1.852)).sum()), 1),
        "fuel_t": round(fuel_kg / 1000, 2),
        "speeds_kn": {f"{v:g}": int(n) for v, n in zip(*np.unique(np.round(speeds_kn, 2), return_counts=True))},
        "links": r.links,
        "seconds": round(seconds, 1),
        "lat": np.round(r.lat, 4).tolist(),
        "lon": np.round(r.lon, 4).tolist(),
        "on_link": r.on_link.tolist(),
        "segment_kn": np.round(speeds_kn, 2).tolist(),
    }


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="The shortest and the least-fuel route of one voyage, arriving by a deadline.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("out", help="result .json")
    parser.add_argument("--start", required=True, help="lat,lon")
    parser.add_argument("--goal", required=True, help="lat,lon")
    parser.add_argument("--via", action="append", default=[], help="lat,lon of a waypoint, passed in the order given; repeat for more")
    parser.add_argument("--departure", required=True, help="UTC, e.g. 2026-10-22T00:00")
    parser.add_argument("--deadline", required=True, help="UTC, e.g. 2026-11-05T12:00")
    parser.add_argument("--speeds-kn", default=",".join(map(str, SPEEDS_KN)), help="speeds to choose from, comma-separated")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
    parser.add_argument("--max-tries", type=int, default=12, help="searches to meet the deadline")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    parser.add_argument("--bathymetry", help="GEBCO grid, netCDF: keep out of water --min-depth m deep or less")
    parser.add_argument("--min-depth", type=float, default=15.0)
    parser.add_argument("--link-approach-km", type=float, default=10.0, help="no depth check this close to a link's sea end")
    parser.add_argument("--graph", help=".npz file to keep the graph in between runs; rebuilt when the options above change")
    parser.add_argument("--legs", choices=["distance", "shared"], default="distance",
                        help="time per leg: in proportion to distance, or one price of an hour for the whole voyage")
    parser.add_argument("--no-smoothing", action="store_true", help="leave sharp turns as found")
    args = parser.parse_args()

    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    if args.bathymetry:
        from shiprouting.sea.depth import Bathymetry, LandOrShallow, link_ends

        land = LandOrShallow(land, Bathymetry(args.bathymetry), args.min_depth, link_ends(), args.link_approach_km)
    t0 = time.perf_counter()
    key = json.dumps({k: getattr(args, k) for k in ("land", "min_step", "links", "bathymetry", "min_depth", "link_approach_km")})
    graph = Graph.cached(args.graph, key, lambda: Graph.build(land, min_step=args.min_step, links=links))
    print(f"graph: {len(graph.lat)} cells, links {[k.name for *_, k in graph.links]}, ready in {time.perf_counter() - t0:.0f} s", flush=True)
    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    start, goal, *via = [tuple(float(v) for v in p.split(",")) for p in (args.start, args.goal, *args.via)]
    speeds = [float(v) for v in args.speeds_kn.split(",")]
    out = plan(graph, KCS_SHIP, forecast, start, goal, args.departure, args.deadline, speeds, args.tables, args.max_tries,
               args.max_seconds, via, args.legs, None if args.no_smoothing else land)
    print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "on_link", "segment_kn", "clock")} if isinstance(v, dict) else v)
           for k, v in out.items() if k != "tries"}, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
