"""Why the oracle sometimes burns more than a plan made on forecasts.

The oracle plans once in the weather that came (ERA5 with currents), known in full; a forecast
plan sailed and judged in that same weather should not beat it, yet some do. This plans those
voyages in the oracle's way again and keeps every route the price-of-an-hour search
(speed.meet_eta) found on the way, in time or late. Each is then finished as plan() finishes
the one it keeps (shortcut, relax, snap_up, smoothed turns, speeds chosen again for the deadline
at the plan's price) and costed in the same weather. If one of them beats the route the oracle
kept, the search found the better way but threw it away; if none comes near the forecast
plan's fuel, the search never went that way.

    python scripts/oracle_gap.py MATRIX LAND --cases busan_la:0828,la_busan:0921 --era5 DIR --cmems DIR
        [--tables DIR] [--out gap.json] [sea options]

A case is route:MMDD, a summer departure of the matrix.
"""

import argparse
import dataclasses
import json
import time

from experiment import departures, truth_weather

import shiprouting.route.plan as plan_module
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route import speed
from shiprouting.route.plan import SPEEDS_KN, add_sea_args, plan, sea_graph
from shiprouting.route.voyage import HOUR
from shiprouting.sea.quadtree import smooth_turns


def finished(choice, land, r, per_hour: float, eta_hours: float, price: float):
    """A route the search found, finished as plan() finishes the one it keeps (no waypoints here). None if it cannot arrive in time."""
    r = dataclasses.replace(r, cost=r.cost - per_hour * float(r.clock[-1]))
    f = choice.shortcut(land, r, [])
    f = choice.snap_up(smooth_turns(land, choice.relax(land, f, []), keep_points=[]))
    try:
        return choice.retime(f, eta_hours, per_hour=price)
    except ValueError:
        return None


def every_route(
    graph,
    land,
    weather,
    start,
    goal,
    departure,
    deadline,
    tables=None,
    max_tries: int = 12,
    max_seconds: float = 600.0,
    speeds_kn=SPEEDS_KN,
) -> dict:
    """Plan as the oracle does, keeping every route the price search found, each finished and costed."""
    found = []
    meet_eta, astar = plan_module.meet_eta, speed.astar

    def keeping(graph, s, g, choice, eta_hours, **kw):
        routes = []

        def search(*a, **k):
            r = astar(*a, **k)
            routes.append((choice.per_hour, r))
            return r

        speed.astar = search
        try:
            f, tries = meet_eta(graph, s, g, choice, eta_hours, **kw)
        finally:
            speed.astar = astar
        found.append((choice, eta_hours, routes))
        return f, tries

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
    ((choice, eta_hours, routes),) = found
    price = p["prices"][0]
    tries = []
    for per_hour, route in routes:
        f = finished(choice, land, route, per_hour, eta_hours, price)
        tries.append(
            {
                "per_hour": round(per_hour, 1),
                "found_h": round(float(route.clock[-1]), 2),
                "found_t": round(
                    (route.cost - per_hour * float(route.clock[-1])) / 1000, 2
                ),
                "found_in_time": bool(route.clock[-1] <= eta_hours * (1 + 1e-9)),
                "km": round(float(route.km), 1),
                "finished_t": None if f is None else round(f.cost / 1000, 2),
                "finished_h": None
                if f is None
                else round(float(f.clock[-1] - f.clock[0]), 2),
            }
        )
    best = min(
        (t["finished_t"] for t in tries if t["finished_t"] is not None), default=None
    )
    return {
        "departure": str(departure),
        "deadline": str(deadline),
        "shortest_t": p["shortest"]["fuel_t"],
        "kept_t": p["least_fuel"]["fuel_t"],
        "plan_seconds": p["least_fuel"]["seconds"],
        "ways_t": p.get("ways_t"),
        "is_shortest": p["least_fuel"].get("is_shortest"),
        "price": price,
        "best_finished_t": best,
        "tries": tries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Every route the oracle's price search found, finished and costed."
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
    parser.add_argument("--out", help=".json of every case's tries")
    add_sea_args(parser)
    args = parser.parse_args()
    with open(args.matrix, encoding="utf-8") as f:
        matrix = json.load(f)
    land, graph = sea_graph(args)
    results = []
    for case in args.cases.split(","):
        name, mmdd = case.split(":")
        r = matrix["routes"][name]
        departure = next(
            d
            for d in departures(matrix["seasons"][args.season])
            if str(d)[5:10].replace("-", "") == mmdd
        )
        deadline = departure + r["hours"] * HOUR
        weather = None  # let the last case's weather go first
        weather = truth_weather(args.era5, args.cmems, departure, deadline)
        t0 = time.perf_counter()
        out = every_route(
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
        results.append(
            {"case": case, "seconds": round(time.perf_counter() - t0, 1)} | out
        )
        print(
            f"{case}: shortest {out['shortest_t']} t, kept {out['kept_t']} t (shortest: {out['is_shortest']}),"
            f" price {out['price']} kg/h, {results[-1]['seconds']:.0f} s",
            flush=True,
        )
        for t in out["tries"]:
            print(
                f"  price {t['per_hour']:8.1f}: found {t['found_h']:7.2f} h {t['found_t']:8.2f} t"
                f" ({'in time' if t['found_in_time'] else 'late'}), finished {t['finished_t']} t",
                flush=True,
            )
        print(
            f"  best finished {out['best_finished_t']} t against kept {out['kept_t']} t",
            flush=True,
        )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
