"""Run the pre-registered scenario matrix (experiments/matrix.json, docs/experiment-plan.md) and
collect the results.

    python scripts/experiment.py deadlines MATRIX LAND --runs DIR [sea options]
    python scripts/experiment.py run MATRIX OUT LAND --season summer [--only busan_la,...] [sea options] [--tables DIR]
        [--oracle ERA5_DIR --cmems DIR]
    python scripts/experiment.py table OUT result.csv
    python scripts/experiment.py recost RESULT.json... --runs DIR [--tables DIR] [--era5 DIR --cmems DIR] [--on-time]
    python scripts/experiment.py backfill RESULT.json... --runs DIR --out DIR [--tables DIR]

deadlines: the routes with no hours yet get the shortest route's distance at 15 kn, rounded up to
12 h (printed, to be written into the matrix before any run).
run: every route x departure x strategy of a season, one JSON each, written as
<route>_<YYYYMMDDHH>_<strategy>.json with the code's commit and the weather used. Results
already there are kept, so a stopped run carries on where it was. The "daily" voyage also
sails the first plan's shortest route, which is the baseline. With --oracle each voyage is instead
planned once in the weather that came (ERA5, with currents), known in full from the start, and
sailed there: the most a plan could save, against which the forecast strategies are measured.
table: one row per route and departure, the strategies side by side.
recost: sail the tracks of saved voyages again in the weather that came (evaluate.recost)
and compare with the fuel they were saved with. With --era5 the weather that came is ERA5 wind
and waves with Copernicus Marine currents (--cmems) instead of the ECMWF forecasts' first hours.
With --on-time each track's speeds are chosen again to arrive by the deadline in that weather
(evaluate.on_time), so the routes are compared at the same arrival time.
backfill: results saved before the shortest route kept its speeds (v25) get them back, so that
recost can sail it: the one speed through the water that, sailed in the weather that came, takes
the hours it was saved with (bisection), and no links (on_link all false: a canal is sailed in
its weather, not at the calm rate). The fuel sailed again at that speed is printed against the
saved fuel as a check. Written to --out under the same names; the originals are left alone.
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
from shiprouting.route.evaluate import Sea, on_time, recost
from shiprouting.route.plan import SPEEDS_KN, add_sea_args, plan, sea_graph
from shiprouting.route.voyage import HOUR, Runs, came_files, voyage
from shiprouting.weather import ecmwf, era5

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


def truth_weather(era5_dir, cmems_dir, departure, deadline):
    """ERA5 (and Copernicus Marine currents) from the day before a voyage leaves to two days after its deadline."""
    days = [str(d).replace("-", "") for d in np.arange(np.datetime64(departure - 24 * HOUR, "D"), np.datetime64(deadline + 48 * HOUR, "D") + 1)]
    currents = [os.path.join(cmems_dir, f"cmems_{d}.npz") for d in days] if cmems_dir else ()
    return era5.load([os.path.join(era5_dir, f"era5_{d}.npz") for d in days], np.float32, currents)


class Known:
    """In place of Runs for the oracle: at every time, the weather that comes, known in full."""

    def __init__(self, weather):
        self.weather = weather

    def newest(self, t):
        return self.weather.time[0], self.weather


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
    if args.oracle:
        strategies = {"oracle": 0}
        meta = {"commit": commit(), "season": args.season, "weather_plan": "the weather that came, known in full (oracle)",
                "weather_came": "era5" + (" + copernicus marine currents" if args.cmems else "")}
    else:
        strategies = matrix["strategies"]
        runs = Runs(args.runs or season["runs"], args.lag_hours)
        meta = {"commit": commit(), "season": args.season, "weather_plan": "ecmwf open data, newest run out",
                "weather_came": "ecmwf open data, steps 0-21 h of each daily run", "lag_hours": args.lag_hours}
    for departure in departures(season):
        for name, r in routes.items():
            deadline = departure + r["hours"] * HOUR
            came = None
            for strategy, every in strategies.items():
                path = os.path.join(args.out, f"{name}_{str(departure)[:13].replace('-', '').replace('T', '')}_{strategy}.json")
                if os.path.exists(path):
                    continue
                if came is None and args.oracle:
                    came = truth_weather(args.oracle, args.cmems, departure, deadline)
                    runs = Known(came)
                elif came is None:
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
    voyage, sea, known = None, None, {}
    for path in sorted(args.results):  # the results of one departure next to each other, sharing the weather
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        departure, deadline = np.datetime64(v["departure"], "s"), np.datetime64(v["deadline"], "s")
        if (departure, deadline) != voyage:
            voyage, sea, weather, known = (departure, deadline), None, None, {}  # let the last weather go before loading the next
            weather = truth_weather(args.era5, args.cmems, departure, deadline) if args.era5 else came_weather(runs, departure, deadline)
            sea = Sea(KCS_SHIP, weather, departure, args.tables)
        again = on_time(v, sea, SPEEDS_KN, known) if args.on_time else recost(v, sea, SPEEDS_KN)
        for name, t in again.items():
            d = 100 * (t["fuel_t"] - v[name]["fuel_t"]) / v[name]["fuel_t"]
            worst = max(worst, abs(d))
            print(f"{os.path.basename(path)} {name}: saved {v[name]['fuel_t']} t {v[name]['hours']} h,"
                  f" again {t['fuel_t']} t {t['hours']} h, late {t['late_h']} h ({d:+.3f}%)", flush=True)
        if {"shortest", "sailed"} <= again.keys():
            was, now = (100 * (w["shortest"]["fuel_t"] - w["sailed"]["fuel_t"]) / w["shortest"]["fuel_t"] for w in (v, again))
            print(f"{os.path.basename(path)} saving over the shortest route: saved {was:+.2f}%, again {now:+.2f}%", flush=True)
    print(f"largest difference {worst:.3f}%")


def cmd_backfill(args) -> None:
    runs = Runs(args.runs)
    os.makedirs(args.out, exist_ok=True)
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        s = v["shortest"]
        if "on_link" not in s:
            departure, deadline = np.datetime64(v["departure"], "s"), np.datetime64(v["deadline"], "s")
            sea = Sea(KCS_SHIP, came_weather(runs, departure, deadline), departure, args.tables)
            lat, lon = np.array(s["lat"]), np.array(s["lon"])
            on_link = np.zeros(len(lat) - 1, bool)

            def sail(kn):
                kg, h, _, _ = sea.sail(lat, lon, on_link, np.full(len(on_link), kn), SPEEDS_KN, 0.0)  # noqa: B023
                return float(h.sum()), float(kg.sum())

            lo, hi = min(SPEEDS_KN), max(SPEEDS_KN)
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if sail(mid)[0] > s["hours"] else (lo, mid)
            kn = round((lo + hi) / 2, 4)
            h, kg = sail(kn)
            s |= {"on_link": on_link.tolist(), "segment_kn": [kn] * len(on_link), "backfilled_kn": kn}
            print(f"{os.path.basename(path)}: shortest at {kn} kn, {h:.1f} h (saved {s['hours']} h), {kg / 1000:.2f} t"
                  f" (saved {s['fuel_t']} t, {100 * (kg / 1000 - s['fuel_t']) / s['fuel_t']:+.3f}%)", flush=True)
        with open(os.path.join(args.out, os.path.basename(path)), "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False)


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
    p.add_argument("--oracle", metavar="ERA5", help="folder of ERA5 days: plan once in the weather that came, known in full, and sail"
                   " there (strategy 'oracle'), instead of the matrix's strategies on the forecasts")
    p.add_argument("--cmems", help="folder of Copernicus Marine current days, with --oracle")
    add_sea_args(p)
    p = sub.add_parser("table")
    p.add_argument("out")
    p.add_argument("csv")
    p = sub.add_parser("recost")
    p.add_argument("results", nargs="+")
    p.add_argument("--runs", required=True)
    p.add_argument("--tables")
    p.add_argument("--era5", help="folder of ERA5 days (weather.era5): sail in these instead of the forecasts")
    p.add_argument("--cmems", help="folder of Copernicus Marine current days (weather.cmems), with --era5")
    p.add_argument("--on-time", action="store_true", help="choose each track's speeds again to arrive by the deadline")
    p = sub.add_parser("backfill")
    p.add_argument("results", nargs="+")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tables")
    args = parser.parse_args()
    if args.cmd in ("deadlines", "run"):
        with open(args.matrix, encoding="utf-8") as f:
            matrix = json.load(f)
        (cmd_deadlines if args.cmd == "deadlines" else cmd_run)(args, matrix)
    else:
        {"table": cmd_table, "recost": cmd_recost, "backfill": cmd_backfill}[args.cmd](args)


if __name__ == "__main__":
    main()
