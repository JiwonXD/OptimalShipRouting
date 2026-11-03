"""Run the pre-registered scenario matrix (experiments/matrix.json, docs/experiment-plan.md) and
collect the results.

    python scripts/experiment.py deadlines MATRIX LAND --runs DIR [sea options]
    python scripts/experiment.py run MATRIX OUT LAND --season summer [--only busan_la,...] [sea options] [--tables DIR]
        [--oracle ERA5_DIR --cmems DIR]
    python scripts/experiment.py distances MATRIX LAND [--pairs a-b,...] [--reference pub151.csv] [--out distances.csv] [sea options]
    python scripts/experiment.py table OUT result.csv
    python scripts/experiment.py recost RESULT.json... --runs DIR [--tables DIR] [--era5 DIR --cmems DIR] [--on-time]
    python scripts/experiment.py backfill RESULT.json... --runs DIR --out DIR [--tables DIR]
    python scripts/experiment.py uncertainty JUDGED.json... --era5 DIR [--cmems DIR] [--tables DIR] --out result.csv
    python scripts/experiment.py sobol result.npz --out result.csv

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
uncertainty: the savings of tracks judged on time (recost --on-time --out) under the fuel model's
uncertain inputs (fuel.uncertainty.PARAMETERS): Monte Carlo spread, how often the sign holds, and
Sobol indices, per route and strategy. The tracks keep their judged speeds; only the ship changes.
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


def cmd_distances(args, matrix) -> None:
    """The shortest route between ports, against a published distance table (WP4: NGA Pub. 151, ±3 %)."""
    from shiprouting.weather.forecast import Forecast

    land, graph = sea_graph(args)
    t = np.datetime64("2026-01-01T00", "s")
    lat, lon = np.arange(-90.0, 90.5, 1.0), np.arange(0.0, 360.0, 1.0)
    names = ("wind_u", "wind_v", "Hs", "wave_from_x", "wave_from_y", "T_m10", "current_u", "current_v")
    still = Forecast(lat, lon, np.array([t, t + 3000 * HOUR]), {n: np.zeros((2, len(lat), len(lon))) for n in names})  # no weather
    reference = {}
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            reference = {(r["from"], r["to"]): r for r in csv.DictReader(f)}
    ports = matrix["ports"]
    pairs = [tuple(p.split("-")) for p in args.pairs.split(",")] if args.pairs else list(reference) or [
        (a, b) for i, a in enumerate(ports) for b in list(ports)[i + 1 :]]
    rows = []
    for a, b in pairs:
        p = plan(graph, KCS_SHIP, still, ports[a], ports[b], t, t + 2000 * HOUR, tables=args.tables, land=land, least_fuel=False)
        if "error" in p:
            print(f"{a}-{b}: {p['error']}", flush=True)
            continue
        km = p["shortest"]["km"]
        row = {"from": a, "to": b, "km": km, "nm": round(km / 1.852, 1), "moved_km": sum(p["moved_km"]),
               "links": int(np.sum(p["shortest"]["on_link"]))}
        if (ref := reference.get((a, b)) or reference.get((b, a))) is not None:
            row |= {"reference_nm": float(ref["nm"]), "via": ref.get("via", ""),
                    "diff_pct": round(100 * (km / 1.852 - float(ref["nm"])) / float(ref["nm"]), 2)}
        print(row, flush=True)
        rows.append(row)
    if args.out and rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


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
    picked = [int(k) for k in args.departures.split(",")] if args.departures else None
    for k, departure in enumerate(departures(season)):
        if picked is not None and k not in picked:
            continue
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
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            route, strategy = route_strategy(path, v)
            judged = {"weather": "era5" + (" + copernicus marine currents" if args.cmems else "") if args.era5 else "ecmwf, steps 0-21 h",
                      "speeds": "chosen again to arrive by the deadline" if args.on_time else "as planned", "commit": commit()}
            with open(os.path.join(args.out, os.path.basename(path)), "w", encoding="utf-8") as f:
                json.dump({k: v[k] for k in ("departure", "deadline", "hours")} | again
                          | {"meta": {"route": route, "strategy": strategy, "judged": judged, "from": os.path.basename(path)}}, f, ensure_ascii=False)
        for name, t in again.items():
            d = 100 * (t["fuel_t"] - v[name]["fuel_t"]) / v[name]["fuel_t"]
            worst = max(worst, abs(d))
            print(f"{os.path.basename(path)} {name}: saved {v[name]['fuel_t']} t {v[name]['hours']} h,"
                  f" again {t['fuel_t']} t {t['hours']} h, late {t['late_h']} h ({d:+.3f}%)", flush=True)
        if {"shortest", "sailed"} <= again.keys():
            was, now = (100 * (w["shortest"]["fuel_t"] - w["sailed"]["fuel_t"]) / w["shortest"]["fuel_t"] for w in (v, again))
            print(f"{os.path.basename(path)} saving over the shortest route: saved {was:+.2f}%, again {now:+.2f}%", flush=True)
    print(f"largest difference {worst:.3f}%")


def route_strategy(path, v) -> tuple[str, str]:
    """A result's route and strategy: from its meta, or from its name (<route>_<departure>_<strategy>.json, v25_ before v25's)."""
    if "route" in v.get("meta", {}):
        return v["meta"]["route"], v["meta"]["strategy"]
    route, _, strategy = os.path.splitext(os.path.basename(path))[0].rsplit("_", 2)
    return route.removeprefix("v25_"), strategy


def track_parts(t: dict, sea: Sea) -> tuple[np.ndarray, ...]:
    """A judged track's speed indices (into SPEEDS_KN), and its moves' air and wave added resistance and hours (links, at the
    calm-water rate: the air resistance of the ship's own speed in still air, no waves, the saved hours)."""
    from shiprouting.fuel.wind import air_resistance
    from shiprouting.route.fuel import KNOT

    lat, lon, on_link = np.array(t["lat"]), np.array(t["lon"]), np.array(t["on_link"], bool)
    kn, h_saved = np.array(t["segment_kn"]), np.array(t["segment_h"])
    clock = np.r_[0.0, np.cumsum(h_saved)]
    k = np.searchsorted(SPEEDS_KN, kn)
    if not np.allclose(np.array(SPEEDS_KN)[np.minimum(k, len(SPEEDS_KN) - 1)], kn):
        raise ValueError("a track's speeds are not all among SPEEDS_KN: judge it on time first (recost --on-time --out)")
    air = np.array([air_resistance(sea.ship.windage, 0.0, 0.0, v * KNOT) for v in kn]) if sea.ship.windage else np.zeros(len(kn))
    wave, h = np.zeros(len(kn)), h_saved.copy()
    for j in np.unique(k):
        i = np.nonzero((k == j) & ~on_link)[0]
        air[i], wave[i], h[i] = sea.cost(SPEEDS_KN[j]).resistance_parts(lat[i], lon[i], lat[i + 1], lon[i + 1], clock[i])
    if not np.allclose(h, h_saved, rtol=1e-4, atol=1e-6):
        raise ValueError("the moves' hours differ from the saved ones: not the weather the track was judged in")
    return k, air, wave, h


def cmd_uncertainty(args) -> None:
    from shiprouting.fuel import uncertainty as u
    from shiprouting.route.fuel import KNOT

    tracks, voyages, voyage, sea, known = [], [], None, None, {}
    for path in sorted(args.results):
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        departure, deadline = np.datetime64(v["departure"], "s"), np.datetime64(v["deadline"], "s")
        if (departure, deadline) != voyage:
            voyage, sea, weather, known = (departure, deadline), None, None, {}
            weather = truth_weather(args.era5, args.cmems, departure, deadline)
            sea = Sea(KCS_SHIP, weather, departure, args.tables)
        index = {}
        for name in ("shortest", "sailed"):
            t = v[name]
            key = (tuple(t["lat"]), tuple(t["lon"]), v["hours"])
            if key not in known:
                known[key] = len(tracks)
                tracks.append(track_parts(t, sea))
            index[name] = known[key]
        voyages.append((os.path.basename(path), *route_strategy(path, v), index["shortest"], index["sailed"],
                        v["shortest"]["fuel_t"], v["sailed"]["fuel_t"]))
        print(f"{os.path.basename(path)}: {len(tracks)} tracks", flush=True)
    sea = weather = None
    A, B, AB = u.saltelli(args.n, np.random.default_rng(args.seed))
    samples = np.vstack([u.NOMINAL[None], A, B, AB.reshape(-1, len(u.PARAMETERS))])
    fuel, over = np.empty((len(samples), len(tracks))), np.zeros(len(samples), int)
    t0 = time.perf_counter()
    for s, p in enumerate(samples):
        tables = u.rate_tables(u.ship_with(KCS_SHIP, p), [kn * KNOT for kn in SPEEDS_KN])
        for j, (k, air, wave, h) in enumerate(tracks):
            fuel[s, j], n_over = u.track_fuel(k, air, wave, h, tables, p[4], p[5])
            over[s] += n_over
        if s % 500 == 0:
            print(f"sample {s}/{len(samples)}, {time.perf_counter() - t0:.0f} s", flush=True)
    saving = np.array([100 * (fuel[:, a] - fuel[:, b]) / fuel[:, a] for _, _, _, a, b, _, _ in voyages]).T  # (samples, voyages)
    nominal_check = max(max(abs(fuel[0, a] / 1000 / fs - 1), abs(fuel[0, b] / 1000 / fp - 1)) for _, _, _, a, b, fs, fp in voyages)
    print(f"nominal sample against the judged fuel: largest difference {100 * nominal_check:.3f}%")
    print(f"moves over MCR, taken at MCR's fuel rate: {int(over[1 : 1 + 2 * args.n].sum())} in {2 * args.n} samples"
          f" ({over[1 : 1 + 2 * args.n].sum() / (2 * args.n * sum(len(t[0]) for t in tracks)):.2e} of moves)")
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", samples=samples, fuel=fuel, saving=saving, over=over,
                        parameters=[p.name for p in u.PARAMETERS], voyages=[x[0] for x in voyages],
                        routes=[x[1] for x in voyages], strategies=[x[2] for x in voyages], n=args.n)
    summarise(saving, [(x[1], x[2]) for x in voyages], args.n, args.out, args.seed)


def summarise(saving, groups_of, n: int, out: str, seed: int) -> None:
    """Per route and strategy (and all): the nominal saving, its Monte Carlo spread and sign kept,
    and Sobol indices with 95 % bootstrap intervals; printed and written to out (.csv). saving:
    (samples, voyages), rows nominal, A (n), B (n), AB (d x n)."""
    from shiprouting.fuel import uncertainty as u

    d = len(u.PARAMETERS)
    mc = slice(1, 1 + 2 * n)  # A and B: independent samples
    groups = {"all": np.ones(len(groups_of), bool)}
    for key in dict.fromkeys(groups_of):
        groups[f"{key[0]} {key[1]}"] = np.array([g == key for g in groups_of])
    rng = np.random.default_rng(seed)
    rows = []
    for g, on in groups.items():
        y = saving[:, on].mean(axis=1)
        parts = (y[1 : 1 + n], y[1 + n : 1 + 2 * n], y[1 + 2 * n :].reshape(d, n))
        S1, ST = u.sobol(*parts)
        (S1_lo, S1_hi), (ST_lo, ST_hi) = u.sobol_interval(*parts, rng)
        sign = np.sign(y[0])
        row = {"group": g, "voyages": int(on.sum()), "nominal_pct": round(float(y[0]), 3),
               "p5_pct": round(float(np.percentile(y[mc], 5)), 3), "p50_pct": round(float(np.percentile(y[mc], 50)), 3),
               "p95_pct": round(float(np.percentile(y[mc], 95)), 3), "sign_kept": round(float(np.mean(np.sign(y[mc]) == sign)), 4),
               "voyages_sign_kept_90": int((np.mean(np.sign(saving[mc][:, on]) == np.sign(saving[0, on]), axis=0) >= 0.9).sum())}  # fmt: skip
        print(row, flush=True)
        for k, p in enumerate(u.PARAMETERS):
            row |= {f"S1_{p.name}": round(float(S1[k]), 3), f"S1_{p.name}_ci": f"{S1_lo[k]:.3f}..{S1_hi[k]:.3f}",
                    f"ST_{p.name}": round(float(ST[k]), 3), f"ST_{p.name}_ci": f"{ST_lo[k]:.3f}..{ST_hi[k]:.3f}"}
            print(f"  {p.name}: S1 {S1[k]:.3f} ({S1_lo[k]:.3f}..{S1_hi[k]:.3f}), ST {ST[k]:.3f} ({ST_lo[k]:.3f}..{ST_hi[k]:.3f})")
        rows.append(row)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


def cmd_sobol(args) -> None:
    with np.load(args.npz) as f:
        names = [str(v) for v in f["voyages"]]
        groups_of = (list(zip(map(str, f["routes"]), map(str, f["strategies"]))) if "routes" in f
                     else [route_strategy(name, {}) for name in names])
        n = int(f["n"]) if "n" in f else (len(f["samples"]) - 1) // (len(f["parameters"]) + 2)
        saving = f["saving"]
    summarise(saving, groups_of, n, args.out, args.seed)


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
    p = sub.add_parser("distances")
    p.add_argument("matrix")
    p.add_argument("land", help="land polygons shapefile")
    p.add_argument("--pairs", help="port pairs from the matrix, e.g. busan-la,rotterdam-newyork (default: the reference's, or all)")
    p.add_argument("--reference", help="CSV of from,to,nm[,via]: published distances to compare with")
    p.add_argument("--out", help=".csv of the distances")
    p.add_argument("--tables", help="folder of wave added resistance tables")
    add_sea_args(p)
    p = sub.add_parser("run")
    p.add_argument("matrix")
    p.add_argument("out", help="folder for the results")
    p.add_argument("land", help="land polygons shapefile")
    p.add_argument("--season", required=True)
    p.add_argument("--only", help="routes, comma-separated")
    p.add_argument("--departures", help="which of the season's departures, comma-separated indices from 0 (to split a run over processes)")
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
    p.add_argument("--out", help="folder to write the judged tracks to, under the results' names")
    p = sub.add_parser("uncertainty")
    p.add_argument("results", nargs="+", help="tracks judged on time in ERA5 (recost --era5 --on-time --out)")
    p.add_argument("--era5", required=True)
    p.add_argument("--cmems")
    p.add_argument("--tables")
    p.add_argument("--n", type=int, default=512, help="Saltelli base samples: n x (parameters + 2) ship models")
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--out", required=True, help=".csv of the results per route and strategy (and an .npz of every sample)")
    p = sub.add_parser("sobol")
    p.add_argument("npz", help="the .npz uncertainty wrote: summarise its samples again")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261103)
    p = sub.add_parser("backfill")
    p.add_argument("results", nargs="+")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tables")
    args = parser.parse_args()
    if args.cmd in ("deadlines", "run", "distances"):
        with open(args.matrix, encoding="utf-8") as f:
            matrix = json.load(f)
        {"deadlines": cmd_deadlines, "run": cmd_run, "distances": cmd_distances}[args.cmd](args, matrix)
    else:
        {"table": cmd_table, "recost": cmd_recost, "backfill": cmd_backfill, "uncertainty": cmd_uncertainty, "sobol": cmd_sobol}[args.cmd](args)


if __name__ == "__main__":
    main()
