"""Plan one voyage: leave at a given time, arrive by a deadline; the shortest route and the
least-fuel one, compared.

The shortest route sails at the one speed that arrives at the deadline (no slower than the
slowest speed offered; if even the fastest is too slow, the deadline cannot be met). The
least-fuel route picks its way and a speed per move from those offered, arriving by the
deadline (speed.meet_eta). Both are costed in the same forecast, in moving weather from the
departure, so the saving is what choosing the way and the speeds is worth.
"""

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
         tables: str | None = None, max_tries: int = 12, max_seconds: float = math.inf) -> dict:
    departure, deadline = np.datetime64(departure, "s"), np.datetime64(deadline, "s")
    if not forecast.time[0] <= departure < forecast.time[-1]:
        raise ValueError(f"departure {departure} is outside the forecast")
    hours = float((deadline - departure) / np.timedelta64(1, "h"))
    s, g = graph.nearest(*start), graph.nearest(*goal)
    out = {
        "start": list(start), "goal": list(goal), "departure": str(departure), "deadline": str(deadline), "hours": hours,
        "moved_km": [round(float(haversine_km(graph.lat[n], graph.lon[n], *p)), 1) for n, p in ((s, start), (g, goal))],
    }

    def cost(kn):
        if tables:
            waves = AddedResistanceTable.cached(tables, ship.seakeeping, kn * KNOT)
        else:
            waves = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        return FuelCost(ship, forecast, departure, kn * KNOT, waves, frozen=False)

    t0 = time.perf_counter()
    r = astar(graph, s, g)
    need_kn = r.km / hours / 1.852
    if need_kn > max(speeds_kn):
        out["error"] = f"the deadline cannot be met: the shortest route ({r.km:.0f} km) needs {need_kn:.1f} kn"
        return out
    kn = max(need_kn, min(speeds_kn))
    fixed = cost(kn)
    out["shortest"] = describe(r, fixed.path(r.lat, r.lon, r.on_link), np.full(len(r.lat) - 1, kn), time.perf_counter() - t0)

    t0 = time.perf_counter()
    # The shortest route's own speed is offered too, so sailing it as it does is among the choices.
    choice = SpeedChoice([fixed if k == kn else cost(k) for k in sorted({*speeds_kn, kn})])
    r, tries = meet_eta(graph, s, g, choice, hours, max_tries=max_tries, max_seconds=max_seconds)
    out["tries"] = tries
    if r is None:
        out["error"] = "no route choosing speeds arrived by the deadline"
        return out
    out["least_fuel"] = describe(r, r.cost, choice.speeds_kn(r), time.perf_counter() - t0)
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
    parser.add_argument("--departure", required=True, help="UTC, e.g. 2026-10-22T00:00")
    parser.add_argument("--deadline", required=True, help="UTC, e.g. 2026-11-05T12:00")
    parser.add_argument("--speeds-kn", default=",".join(map(str, SPEEDS_KN)), help="speeds to choose from, comma-separated")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
    parser.add_argument("--max-tries", type=int, default=12, help="searches to meet the deadline")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
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
    point = [tuple(float(v) for v in p.split(",")) for p in (args.start, args.goal)]
    speeds = [float(v) for v in args.speeds_kn.split(",")]
    out = plan(graph, KCS_SHIP, forecast, *point, args.departure, args.deadline, speeds, args.tables, args.max_tries, args.max_seconds)
    print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "on_link", "segment_kn")} if isinstance(v, dict) else v)
           for k, v in out.items() if k != "tries"}, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
