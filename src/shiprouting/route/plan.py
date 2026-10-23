"""Plan one voyage: leave at a given time, arrive by a deadline; the shortest route and the
least-fuel one, compared.

The shortest route sails at the one speed that arrives at the deadline (no slower than the
slowest speed offered; if even the fastest is too slow, the deadline cannot be met). The
least-fuel route picks its way and a speed per move from those offered, arriving by the
deadline (speed.meet_eta). Both are costed in the same forecast, in moving weather from the
departure, so the saving is what choosing the way and the speeds is worth.

Waypoints are passed in order. Each leg is searched on its own and the legs are chained.
The least-fuel route gives each leg a share of the time in proportion to its shortest
distance, and sets out on the next leg when the last one arrives.
"""

import dataclasses
import math
import time

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.speed import SpeedChoice, meet_eta
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph, astar
from shiprouting.weather.forecast import Forecast

SPEEDS_KN = (12, 14, 16, 18, 20, 22, 24)


def plan(graph: Graph, ship: Ship, forecast: Forecast, start, goal, departure, deadline, speeds_kn=SPEEDS_KN,
         tables: str | None = None, max_tries: int = 12, max_seconds: float = math.inf, via=()) -> dict:
    departure, deadline = np.datetime64(departure, "s"), np.datetime64(deadline, "s")
    if not forecast.time[0] <= departure < forecast.time[-1]:
        raise ValueError(f"departure {departure} is outside the forecast")
    hours = float((deadline - departure) / np.timedelta64(1, "h"))
    points = [start, *via, goal]
    nodes = [graph.nearest(*p) for p in points]
    out = {
        "start": list(start), "goal": list(goal), "via": [list(p) for p in via],
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

    t0 = time.perf_counter()
    legs = [astar(graph, a, b) for a, b in zip(nodes[:-1], nodes[1:])]
    r = chain(legs)
    need_kn = r.km / hours / 1.852
    if need_kn > max(speeds_kn):
        out["error"] = f"the deadline cannot be met: the shortest route ({r.km:.0f} km) needs {need_kn:.1f} kn"
        return out
    kn = max(need_kn, min(speeds_kn))
    fixed = cost(kn)
    out["shortest"] = describe(r, fixed.path(r.lat, r.lon, r.on_link), np.full(len(r.lat) - 1, kn), time.perf_counter() - t0)

    t0 = time.perf_counter()
    # The shortest route's own speed is offered too, so sailing it as it does is among the choices.
    offered = sorted({*speeds_kn, kn})
    found, out["tries"], out["legs"], at = [], [], [], departure
    for (a, b), leg in zip(zip(nodes[:-1], nodes[1:]), legs):
        choice = SpeedChoice([cost(k, at) for k in offered])
        given = hours * leg.km / r.km
        f, tries = meet_eta(graph, a, b, choice, given, max_tries=max_tries, max_seconds=max_seconds)
        out["tries"].append(tries)
        if f is None:
            out["error"] = "no route choosing speeds arrived by the deadline"
            return out
        found.append(f)
        out["legs"].append({"shortest_km": round(leg.km, 1), "hours_given": round(given, 1), "km": round(f.km, 1),
                            "hours": round(float(f.clock[-1]), 1), "fuel_t": round(f.cost / 1000, 2)})
        at = at + np.timedelta64(int(round(f.clock[-1] * 3600)), "s")
    f = chain(found)
    out["least_fuel"] = describe(f, f.cost, choice.speeds_kn(f), time.perf_counter() - t0)
    saving = out["shortest"]["fuel_t"] - out["least_fuel"]["fuel_t"]
    out["saving_t"] = round(saving, 2)
    out["saving_pct"] = round(100 * saving / out["shortest"]["fuel_t"], 2)
    return out


def chain(legs: list[SearchResult]) -> SearchResult:
    """The legs as one path, each leg's clock carrying on from where the last one ended."""
    clock = [legs[0].clock]
    for r in legs[1:]:
        clock.append(r.clock[1:] - r.clock[0] + clock[-1][-1])
    return dataclasses.replace(
        legs[0],
        lat=np.concatenate([legs[0].lat] + [r.lat[1:] for r in legs[1:]]),
        lon=np.concatenate([legs[0].lon] + [r.lon[1:] for r in legs[1:]]),
        on_link=np.concatenate([r.on_link for r in legs]),
        links=[name for r in legs for name in r.links],
        km=sum(r.km for r in legs),
        expanded=sum(r.expanded for r in legs),
        seconds=sum(r.seconds for r in legs),
        cost=sum(r.cost for r in legs),
        clock=np.concatenate(clock),
    )


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
               args.max_seconds, via)
    print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "on_link", "segment_kn")} if isinstance(v, dict) else v)
           for k, v in out.items() if k not in ("tries", "legs")}, flush=True)
    for leg in out.get("legs", []):
        print(" ", leg, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
