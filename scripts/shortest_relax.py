"""Can the weather improve on the shortest route where the oracle keeps it as it is?

Where the oracle's plan keeps the shortest route (O = S), the price search found no way that
beats it. The shortest route is now searched along great-circle arcs (해결 48), the least-fuel
search still on the cells' steps; this plans those voyages in the oracle's way again, then
moves the shortest route's points sideways where that is cheaper in the same weather
(SpeedChoice.relax, at the shortest route's one speed), chooses the speeds again for the
deadline (retime, at the plan's price), and costs it. If that beats the route the oracle kept,
the weather had a better way next to the shortest route that the search never reached.

    python scripts/shortest_relax.py MATRIX LAND --cases busan_la:0831,busan_la:0918 --era5 DIR --cmems DIR
        [--tables DIR] [--out relax.json] [sea options]

A case is route:MMDD, a summer departure of the matrix.
"""

import argparse
import dataclasses
import json
import time

import numpy as np
from experiment import departures, truth_weather

import shiprouting.route.plan as plan_module
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.fuel import KNOT
from shiprouting.route.plan import SPEEDS_KN, add_sea_args, plan, sea_graph
from shiprouting.route.voyage import HOUR
from shiprouting.sea.latlon import SearchResult


def crossing(lat, lon, at: float = 140.0):
    """The latitude where a track first crosses the meridian at, or None."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    for i in np.nonzero((lon[:-1] - at) * (lon[1:] - at) <= 0)[0]:
        if abs(lon[i + 1] - lon[i]) < 180:
            f = (at - lon[i]) / (lon[i + 1] - lon[i]) if lon[i + 1] != lon[i] else 0.0
            return round(float(lat[i] + f * (lat[i + 1] - lat[i])), 3)
    return None


def relaxed_shortest(
    graph,
    land,
    weather,
    start,
    goal,
    departure,
    deadline,
    tables=None,
    max_tries=12,
    max_seconds=600.0,
    speeds_kn=SPEEDS_KN,
) -> dict:
    """Plan as the oracle does, then the shortest route relaxed in the same weather and retimed."""
    kept = []
    meet_eta = plan_module.meet_eta

    def keeping(graph, s, g, choice, eta_hours, **kw):
        kept.append(choice)
        return meet_eta(graph, s, g, choice, eta_hours, **kw)

    plan_module.meet_eta = keeping
    try:
        p = plan(
            graph,
            KCS_SHIP,
            weather,
            start,
            goal,
            departure,
            deadline,
            speeds_kn,
            tables,
            max_tries,
            max_seconds,
            [],
            land=land,
        )
    finally:
        plan_module.meet_eta = meet_eta
    (choice,) = kept
    hours = p["hours"]
    price = p["prices"][0]
    s = p["shortest"]
    lat, lon, on_link = np.array(s["lat"]), np.array(s["lon"]), np.array(s["on_link"], bool)
    k = int(np.argmin([abs(c.speed / KNOT - s["segment_kn"][0]) for c in choice.costs]))
    r = SearchResult(lat, lon, on_link, s["links"], s["km"], 0, 0.0, 0.0, np.zeros(len(lat)))
    clock, kg, _ = choice.sail(r, np.full(len(lat) - 1, k))
    r = dataclasses.replace(r, clock=clock, cost=float(kg[k].sum()))
    t0 = time.perf_counter()
    relaxed = choice.relax(land, r, [])
    relax_s = time.perf_counter() - t0
    out = {
        "departure": str(departure),
        "shortest_t": s["fuel_t"],
        "kept_t": p["least_fuel"]["fuel_t"],
        "kept_is_shortest": p["least_fuel"].get("is_shortest"),
        "kept_km": p["least_fuel"]["km"],
        "kept_140e": crossing(p["least_fuel"]["lat"], p["least_fuel"]["lon"]),
        "shortest_km": s["km"],
        "shortest_140e": crossing(lat, lon),
        "relaxed_km": round(relaxed.km, 1),
        "relaxed_140e": crossing(relaxed.lat, relaxed.lon),
        "relax_seconds": round(relax_s, 1),
        "price": price,
        "plan_seconds": p["least_fuel"]["seconds"],
    }
    for name, way in (("shortest_retimed_t", r), ("relaxed_retimed_t", choice.snap_up(relaxed))):
        try:
            out[name] = round(choice.retime(way, hours, per_hour=price).cost / 1000, 2)
        except ValueError:
            out[name] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The shortest route relaxed in the oracle's weather, against the oracle."
    )
    parser.add_argument("matrix")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("--cases", required=True, help="route:MMDD, comma-separated")
    parser.add_argument("--season", default="summer")
    parser.add_argument("--era5", required=True)
    parser.add_argument("--cmems")
    parser.add_argument("--tables")
    parser.add_argument("--max-tries", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--out", help=".json of every case")
    add_sea_args(parser)
    args = parser.parse_args()
    with open(args.matrix, encoding="utf-8") as f:
        matrix = json.load(f)
    land, graph = sea_graph(args)
    results = []
    for case in args.cases.split(","):
        name, mmdd = case.split(":")
        r = matrix["routes"][name]
        departure = next(d for d in departures(matrix["seasons"][args.season]) if str(d)[5:10].replace("-", "") == mmdd)
        deadline = departure + r["hours"] * HOUR
        weather = None  # let the last case's weather go first
        weather = truth_weather(args.era5, args.cmems, departure, deadline)
        t0 = time.perf_counter()
        out = relaxed_shortest(
            graph,
            land,
            weather,
            matrix["ports"][r["start"]],
            matrix["ports"][r["goal"]],
            departure,
            deadline,
            args.tables,
            args.max_tries,
            args.max_seconds,
        )
        results.append({"case": case, "seconds": round(time.perf_counter() - t0, 1)} | out)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
