"""Sail a voyage as a ship would: plan at departure, then every day plan the rest again from
where the ship has got to, on the newest forecast run out by then, and sail in the weather
that came.

The weather that came is pieced together from the first 21 h of each daily run
(came_files). A run counts as out lag_hours after it starts: ECMWF open data publishes a
00z run some 7–8 h later, so a ship leaving at 00z plans on the day before's run.

The ship sails each plan's segments at their planned speeds through the water, in the
weather and current that came (evaluate.Sea). Where that needs more than the engine's MCR,
it takes the fastest slower speed it can make. A plan is followed up to its first point, at or after
the next time to replan, that is a cell of the graph and not inside a link; the next plan
starts there, for the waypoints still ahead and the same deadline. When a plan can no
longer meet the deadline, its deadline is moved out until it can, and the voyage is late.
"""

import glob
import math
import os
import re
import time

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.route.evaluate import Sea, track
from shiprouting.route.plan import SPEEDS_KN, plan
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import Graph
from shiprouting.weather import ecmwf
from shiprouting.weather.forecast import Forecast

HOUR = np.timedelta64(3600, "s")


def came_files(folders, last_step: int = 21) -> list[str]:
    """The GRIB2 files of each run folder up to last_step hours: the weather that came."""
    return [p for d in folders for p in sorted(glob.glob(os.path.join(d, "*.grib2")))
            if int(re.search(r"step(\d+)", os.path.basename(p)).group(1)) <= last_step]


class Runs:
    """The forecast runs kept in folders named YYYYMMDDHH (matching `name`) under root; the newest
    out by a time, loaded when first asked for (one at a time)."""

    def __init__(self, root: str, lag_hours: float = 8.0, dtype=np.float32, name: str = r"\d{10}"):
        self.root, self.lag, self.dtype = root, lag_hours, dtype
        names = sorted(n for n in os.listdir(root) if re.fullmatch(name, n))
        self.times = [np.datetime64(f"{n[:4]}-{n[4:6]}-{n[6:8]}T{n[8:10]}", "s") for n in names]
        self.folders = [os.path.join(root, n) for n in names]
        self._run, self._forecast = None, None

    def newest(self, t) -> tuple[np.datetime64, Forecast]:
        out = [k for k, run in enumerate(self.times) if run + self.lag * HOUR <= t]
        if not out:
            raise ValueError(f"no run out by {t}")
        k = out[-1]
        if self._run != self.times[k]:
            self._run, self._forecast = None, None  # let the last one go first
            self._forecast = ecmwf.load(sorted(glob.glob(os.path.join(self.folders[k], "*.grib2"))), self.dtype)
            self._run = self.times[k]
        return self._run, self._forecast


def voyage(graph: Graph, ship: Ship, land, runs: Runs, came: Forecast, start, goal, departure, deadline, via=(),
           speeds_kn=SPEEDS_KN, tables: str | None = None, every_hours: float = 24.0, max_tries: int = 12,
           max_seconds: float = math.inf, last_hours: float = 6.0) -> dict:
    """The least-fuel voyage replanned every every_hours (math.inf: planned once), sailed in the
    weather that came; and the first plan's shortest route sailed there too.

    A replan that would fall within last_hours of the deadline is brought forward to last_hours
    before it (or left out, if the last one was hardly earlier): sailing in the weather that came
    slips from the plan by a few percent a day, and an hour or so out there is no time left to make
    it up. A plan that cannot arrive by the deadline pushes it back a little at a time, from 0.1 h
    or 5 % of the time left, doubling."""
    departure, deadline = np.datetime64(departure, "s"), np.datetime64(deadline, "s")
    hours = float((deadline - departure) / HOUR)
    sea = Sea(ship, came, departure, tables)
    pos, ahead, at = tuple(start), list(via), 0.0
    goal_cell = graph.nearest(*goal)
    sailed_track = {"lat": [], "lon": [], "on_link": [], "planned_kn": [], "segment_kn": [], "kg": [], "hours": []}
    log, first, late, prices = [], None, False, None
    while True:
        now = departure + int(round(at * 3600)) * np.timedelta64(1, "s")
        run, forecast = runs.newest(now)
        t0 = time.perf_counter()
        target = deadline
        for attempt in range(8):
            p = plan(graph, ship, forecast, pos, goal, now, target, speeds_kn, tables, max_tries, max_seconds, ahead, land=land, prices=prices,
                     start_exact=bool(log))  # replanned from where the last plan was cut, not from its nearest cell
            if "least_fuel" in p:
                break
            late = True
            target = target + max(0.05 * float((target - now) / HOUR), 0.1 * 2**attempt) * HOUR
        else:
            raise RuntimeError(f"no plan from {pos} at {now}: {p.get('error')}")
        first = first or p
        f = p["least_fuel"]
        lat, lon = np.array(f["lat"]), np.array(f["lon"])
        on_link, kn = np.array(f["on_link"], bool), np.array(f["segment_kn"])
        stops = np.array([False] + [not on_link[j - 1] and not on_link[j] for j in range(1, len(lat) - 1)] + [False])
        stops[stops] = [graph.nearest(a, b) != goal_cell for a, b in zip(lat[stops], lon[stops])]  # a plan needs two cells
        until = at + every_hours
        if math.isfinite(until) and until > hours - last_hours:  # too close to the deadline to make up a slip from there
            until = hours - last_hours if hours - last_hours >= at + last_hours else math.inf
        kg, h, sailed, end = sea.sail(lat, lon, on_link, kn, speeds_kn, at, until, stops)
        seg_km = haversine_km(lat[:end], lon[:end], lat[1 : end + 1], lon[1 : end + 1])
        log.append({
            "time": str(now), "run": str(run), "from": [round(float(v), 4) for v in pos], "deadline": str(target),
            "planned_t": f["fuel_t"], "planned_h": f["hours"], "seconds": round(time.perf_counter() - t0, 1),
            "tries": sum(len(t) for t in p["tries"]),
            "sailed_km": round(float(seg_km.sum()), 1), "sailed_h": round(float(h.sum()), 1), "sailed_t": round(float(kg.sum()) / 1000, 2),
        })  # fmt: skip
        print(f"  {log[-1]}", flush=True)
        t = sailed_track
        if not t["lat"]:
            t["lat"].append(float(lat[0]))
            t["lon"].append(float(lon[0]))
        t["lat"] += lat[1 : end + 1].tolist()
        t["lon"] += lon[1 : end + 1].tolist()
        t["on_link"] += on_link[:end].tolist()
        t["planned_kn"] += kn[:end].tolist()
        t["segment_kn"] += sailed.tolist()
        t["kg"] += kg.tolist()
        t["hours"] += h.tolist()
        at += float(h.sum())
        legs_km = np.cumsum([leg["km"] for leg in f["legs"]])[:-1]
        passed = int((legs_km <= seg_km.sum() + 0.01).sum())
        ahead, prices = ahead[passed:], p["prices"][passed:]  # the next plan starts from this one's prices
        if end == len(lat) - 1:
            break
        pos = (float(lat[end]), float(lon[end]))

    s = first["shortest"]
    s_lat, s_lon, s_link, s_kn = np.array(s["lat"]), np.array(s["lon"]), np.array(s["on_link"], bool), np.array(s["segment_kn"])
    s_kg, s_h, s_sailed, _ = sea.sail(s_lat, s_lon, s_link, s_kn, speeds_kn, 0.0)
    t = {k: np.array(v) for k, v in sailed_track.items()}
    return {
        "departure": str(departure), "deadline": str(deadline), "hours": hours, "every_hours": every_hours,
        "planned": {"shortest_t": s["fuel_t"], "least_fuel_t": first["least_fuel"]["fuel_t"]},
        "shortest": track(s_lat, s_lon, s_link, s_kn, s_kg, s_h, s_sailed, hours),
        "sailed": track(t["lat"], t["lon"], t["on_link"], t["planned_kn"], t["kg"], t["hours"], t["segment_kn"], hours)
        | {"plans": len(log), "deadline_moved": late, "log": log},
    }  # fmt: skip


def main() -> None:
    import argparse
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.route.plan import add_sea_args, sea_graph

    parser = argparse.ArgumentParser(description="A voyage replanned every day on the newest forecast, sailed in the weather that came.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("runs", help="folder holding the forecast runs, one YYYYMMDDHH folder each")
    parser.add_argument("out", help="result .json")
    parser.add_argument("--start", required=True, help="lat,lon")
    parser.add_argument("--goal", required=True, help="lat,lon")
    parser.add_argument("--via", action="append", default=[], help="lat,lon of a waypoint, passed in the order given; repeat for more")
    parser.add_argument("--departure", required=True, help="UTC, e.g. 2026-09-01T00:00")
    parser.add_argument("--deadline", required=True, help="UTC")
    parser.add_argument("--speeds-kn", default=",".join(map(str, SPEEDS_KN)), help="speeds to choose from, comma-separated")
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
    parser.add_argument("--max-tries", type=int, default=12, help="searches to meet the deadline")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    parser.add_argument("--every-hours", type=float, default=24.0, help="replan this often; 0: plan once at departure")
    parser.add_argument("--lag-hours", type=float, default=8.0, help="a run is out this long after it starts")
    add_sea_args(parser)
    args = parser.parse_args()

    land, graph = sea_graph(args)
    runs = Runs(args.runs, args.lag_hours)
    departure, deadline = np.datetime64(args.departure, "s"), np.datetime64(args.deadline, "s")
    came_runs = [d for t, d in zip(runs.times, runs.folders) if departure - 24 * HOUR <= t <= deadline + 48 * HOUR]
    t0 = time.perf_counter()
    came = ecmwf.load(came_files(came_runs), np.float32)
    print(f"weather that came: {came.time[0]} to {came.time[-1]}, loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    start, goal, *via = [tuple(float(v) for v in p.split(",")) for p in (args.start, args.goal, *args.via)]
    speeds = [float(v) for v in args.speeds_kn.split(",")]
    out = voyage(graph, KCS_SHIP, land, runs, came, start, goal, departure, deadline, via, speeds, args.tables,
                 args.every_hours or math.inf, args.max_tries, args.max_seconds)
    print({k: ({kk: vv for kk, vv in v.items() if not isinstance(vv, list)} if isinstance(v, dict) else v)
           for k, v in out.items()}, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
