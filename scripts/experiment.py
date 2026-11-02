"""Run the pre-registered scenario matrix (experiments/matrix.json, docs/experiment-plan.md) and
collect the results.

    python scripts/experiment.py deadlines MATRIX LAND --runs DIR [sea options]
    python scripts/experiment.py run MATRIX OUT LAND --season summer [--only busan_la,...] [sea options] [--tables DIR]
    python scripts/experiment.py table OUT result.csv
    python scripts/experiment.py recost RESULT.json... --runs DIR [--tables DIR]

deadlines: the routes with no hours yet get the shortest route's distance at 15 kn, rounded up to
12 h (printed, to be written into the matrix before any run).
run: every route x departure x strategy of a season, one JSON each, written as
<route>_<YYYYMMDDHH>_<strategy>.json with the code's commit and the weather used. Results
already there are kept, so a stopped run carries on where it was. The "daily" voyage also
sails the first plan's shortest route, which is the baseline.
table: one row per route and departure, the strategies side by side.
recost: sail the tracks of saved voyages again in the weather that came (evaluate.recost)
and compare with the fuel they were saved with.
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.evaluate import Sea, recost
from shiprouting.route.plan import SPEEDS_KN, add_sea_args, plan, sea_graph
from shiprouting.route.voyage import HOUR, Runs, came_files, voyage
from shiprouting.weather import ecmwf

DEADLINE_KN = 15.0


def commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True, text=True, check=False).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def departures(season: dict) -> list[np.datetime64]:
    first = np.datetime64(season["first"], "s")
    return [first + k * season["every_days"] * 24 * HOUR for k in range(season["count"])]


def came_weather(runs: Runs, departure, deadline):
    """The weather that came over a voyage: the first 21 h of each run from the day before it leaves to two days after its deadline."""
    folders = [d for t, d in zip(runs.times, runs.folders) if departure - 24 * HOUR <= t <= deadline + 48 * HOUR]
    return ecmwf.load(came_files(folders), np.float32)


def cmd_deadlines(args, matrix) -> None:
    land, graph = sea_graph(args)
    runs = Runs(args.runs)
    t, forecast = runs.newest(runs.times[0] + 24 * HOUR)
    for name, r in matrix["routes"].items():
        if r["hours"] is not None:
            continue
        start, goal = matrix["ports"][r["start"]], matrix["ports"][r["goal"]]
        p = plan(graph, KCS_SHIP, forecast, start, goal, t, t + 2000 * HOUR, land=land, least_fuel=False)
        km = p["shortest"]["km"]
        hours = math.ceil(km / 1.852 / DEADLINE_KN / 12) * 12
        print(f"{name}: shortest {km:,.1f} km -> {hours} h at {DEADLINE_KN:g} kn", flush=True)


def cmd_run(args, matrix) -> None:
    season = matrix["seasons"][args.season]
    routes = {k: v for k, v in matrix["routes"].items() if not args.only or k in args.only.split(",")}
    if missing := [k for k, v in routes.items() if v["hours"] is None]:
        sys.exit(f"routes with no deadline yet: {missing} (run `deadlines` and write them into the matrix)")
    os.makedirs(args.out, exist_ok=True)
    land, graph = sea_graph(args)
    runs = Runs(args.runs or season["runs"], args.lag_hours)
    meta = {"commit": commit(), "season": args.season, "weather_plan": "ecmwf open data, newest run out",
            "weather_came": "ecmwf open data, steps 0-21 h of each daily run", "lag_hours": args.lag_hours}
    for departure in departures(season):
        for name, r in routes.items():
            deadline = departure + r["hours"] * HOUR
            came = None
            for strategy, every in matrix["strategies"].items():
                path = os.path.join(args.out, f"{name}_{str(departure)[:13].replace('-', '').replace('T', '')}_{strategy}.json")
                if os.path.exists(path):
                    continue
                if came is None:
                    came = came_weather(runs, departure, deadline)
                print(f"{name} {departure} {strategy}", flush=True)
                t0 = time.perf_counter()
                start, goal = matrix["ports"][r["start"]], matrix["ports"][r["goal"]]
                out = voyage(graph, KCS_SHIP, land, runs, came, start, goal, departure, deadline, (), SPEEDS_KN, args.tables,
                             every or math.inf, args.max_tries, args.max_seconds)
                out["meta"] = meta | {"route": name, "strategy": strategy, "seconds": round(time.perf_counter() - t0, 1)}
                with open(path + ".part", "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False)
                os.replace(path + ".part", path)
                print(f"  shortest {out['shortest']['fuel_t']} t, sailed {out['sailed']['fuel_t']} t, late {out['sailed']['late_h']} h,"
                      f" {out['meta']['seconds']:.0f} s", flush=True)


def cmd_table(args) -> None:
    rows = {}
    for path in sorted(glob.glob(os.path.join(args.out, "*.json"))):
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        m = v["meta"]
        row = rows.setdefault((m["route"], v["departure"]), {"route": m["route"], "season": m["season"], "departure": v["departure"],
                                                              "hours": v["hours"], "commit": m["commit"]})
        if m["strategy"] == "daily":
            row |= {"shortest_t": v["shortest"]["fuel_t"], "shortest_late_h": v["shortest"]["late_h"]}
        row |= {f"{m['strategy']}_t": v["sailed"]["fuel_t"], f"{m['strategy']}_late_h": v["sailed"]["late_h"],
                f"{m['strategy']}_plans": v["sailed"]["plans"]}
    for row in rows.values():
        for s in ("once", "daily"):
            if f"{s}_t" in row and "shortest_t" in row:
                row[f"{s}_saving_pct"] = round(100 * (row["shortest_t"] - row[f"{s}_t"]) / row["shortest_t"], 3)
    fields = list(dict.fromkeys(k for row in rows.values() for k in row))
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows.values())
    print(f"{len(rows)} rows -> {args.csv}")


def cmd_recost(args) -> None:
    runs = Runs(args.runs)
    worst = 0.0
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        departure, deadline = np.datetime64(v["departure"], "s"), np.datetime64(v["deadline"], "s")
        sea = Sea(KCS_SHIP, came_weather(runs, departure, deadline), departure, args.tables)
        again = recost(v, sea, SPEEDS_KN)
        for name, t in again.items():
            d = 100 * (t["fuel_t"] - v[name]["fuel_t"]) / v[name]["fuel_t"]
            worst = max(worst, abs(d))
            print(f"{os.path.basename(path)} {name}: saved {v[name]['fuel_t']} t {v[name]['hours']} h,"
                  f" again {t['fuel_t']} t {t['hours']} h ({d:+.3f}%)", flush=True)
    print(f"largest difference {worst:.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-registered scenario matrix and collect the results.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deadlines")
    p.add_argument("matrix")
    p.add_argument("land", help="land polygons shapefile")
    p.add_argument("--runs", required=True)
    add_sea_args(p)
    p = sub.add_parser("run")
    p.add_argument("matrix")
    p.add_argument("out", help="folder for the results")
    p.add_argument("land", help="land polygons shapefile")
    p.add_argument("--season", required=True)
    p.add_argument("--only", help="routes, comma-separated")
    p.add_argument("--runs", help="forecast runs folder, if not the season's")
    p.add_argument("--tables", help="folder of wave added resistance tables")
    p.add_argument("--lag-hours", type=float, default=8.0)
    p.add_argument("--max-tries", type=int, default=12)
    p.add_argument("--max-seconds", type=float, default=600.0)
    add_sea_args(p)
    p = sub.add_parser("table")
    p.add_argument("out")
    p.add_argument("csv")
    p = sub.add_parser("recost")
    p.add_argument("results", nargs="+")
    p.add_argument("--runs", required=True)
    p.add_argument("--tables")
    args = parser.parse_args()
    if args.cmd in ("deadlines", "run"):
        with open(args.matrix, encoding="utf-8") as f:
            matrix = json.load(f)
        (cmd_deadlines if args.cmd == "deadlines" else cmd_run)(args, matrix)
    else:
        (cmd_table if args.cmd == "table" else cmd_recost)(args)


if __name__ == "__main__":
    main()
