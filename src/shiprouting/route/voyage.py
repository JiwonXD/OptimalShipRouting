"""Sail a voyage as a ship would: plan at departure, then every day plan the rest again from
where the ship has got to, on the newest forecast run out by then, and sail in the weather
that came.

The weather that came is pieced together from the first 21 h of each daily run
(came_files). A run counts as out lag_hours after it starts: ECMWF open data publishes a
00z run some 7–8 h later, so a ship leaving at 00z plans on the day before's run.

The ship sails each plan's segments at their planned speeds through the water, in the
weather and current that came. Where that needs more than the engine's MCR, it takes the
fastest slower speed it can make. A plan is followed up to its first point, at or after
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
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
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
    """The forecast runs kept in folders named YYYYMMDDHH under root; the newest out by a time,
    loaded when first asked for (one at a time)."""

    def __init__(self, root: str, lag_hours: float = 8.0, dtype=np.float32):
        self.root, self.lag, self.dtype = root, lag_hours, dtype
        names = sorted(n for n in os.listdir(root) if re.fullmatch(r"\d{10}", n))
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


class Came:
    """Sailing in the weather that came, at given speeds through the water."""

    def __init__(self, ship: Ship, came: Forecast, departure, tables: str | None = None):
        self.ship, self.came, self.departure, self.tables = ship, came, np.datetime64(departure, "s"), tables
        self._costs = {}

    def cost(self, kn: float) -> FuelCost:
        if kn not in self._costs:
            speed = kn * KNOT
            waves = (AddedResistanceTable.between(self.tables, self.ship.seakeeping, speed) if self.tables
                     else AddedResistanceTable.build(self.ship.seakeeping, speed))
            self._costs[kn] = FuelCost(self.ship, self.came, self.departure, speed, waves, frozen=False, currents=True)
        return self._costs[kn]

    def sail(self, lat, lon, on_link, kn, offered, clock0: float, until: float = math.inf, stops=None):
        """Sail the segments from clock0 (hours since departure) at speeds kn, each falling back to
        the fastest slower speed (of kn's and offered) where the engine cannot make it. Stops at the
        first point j with stops[j] reached at or after until. Returns the fuel, kg, hours and speed
        of each segment sailed, and the index of the point it stopped at."""
        offered = sorted({*np.round(kn, 6), *offered})
        seg_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        kg, hours, sailed = [], [], []
        clock = clock0
        for i in range(len(seg_km)):
            for v in [k for k in offered if k <= round(kn[i], 6)][::-1]:
                c = self.cost(v)
                if on_link[i]:
                    kg_i, h_i = seg_km[i] * c.calm_kg_per_km, seg_km[i] / (c.speed * 3.6)
                else:
                    kg_i, h_i = (float(x) for x in c.kg_hours(lat[i], lon[i], lat[i + 1], lon[i + 1], clock))
                if math.isfinite(kg_i):
                    break
            kg.append(kg_i)
            hours.append(h_i)
            sailed.append(v)
            clock += h_i
            if clock >= until and stops is not None and stops[i + 1]:
                return np.array(kg), np.array(hours), np.array(sailed), i + 1
        return np.array(kg), np.array(hours), np.array(sailed), len(seg_km)


def voyage(graph: Graph, ship: Ship, land, runs: Runs, came: Forecast, start, goal, departure, deadline, via=(),
           speeds_kn=SPEEDS_KN, tables: str | None = None, every_hours: float = 24.0, max_tries: int = 12,
           max_seconds: float = math.inf) -> dict:
    """The least-fuel voyage replanned every every_hours (math.inf: planned once), sailed in the
    weather that came; and the first plan's shortest route sailed there too."""
    departure, deadline = np.datetime64(departure, "s"), np.datetime64(deadline, "s")
    hours = float((deadline - departure) / HOUR)
    sea = Came(ship, came, departure, tables)
    cells = {(round(float(a), 6), round(float(b), 6)) for a, b in zip(graph.lat, graph.lon)}
    pos, ahead, at = tuple(start), list(via), 0.0
    track = {"lat": [], "lon": [], "on_link": [], "segment_kn": [], "kg": [], "hours": []}
    log, first, late, prices = [], None, False, None
    while True:
        now = departure + int(round(at * 3600)) * np.timedelta64(1, "s")
        run, forecast = runs.newest(now)
        t0 = time.perf_counter()
        target = deadline
        for _ in range(8):
            p = plan(graph, ship, forecast, pos, goal, now, target, speeds_kn, tables, max_tries, max_seconds, ahead, land=land, prices=prices)
            if "least_fuel" in p:
                break
            late = True
            target = target + max(0.05 * float((target - now) / HOUR), 1.0) * HOUR
        else:
            raise RuntimeError(f"no plan from {pos} at {now}: {p.get('error')}")
        first = first or p
        f = p["least_fuel"]
        lat, lon = np.array(f["lat"]), np.array(f["lon"])
        on_link, kn = np.array(f["on_link"], bool), np.array(f["segment_kn"])
        stops = np.array([False] + [(round(lat[j], 6), round(lon[j], 6)) in cells and not on_link[j - 1] and not on_link[j]
                                    for j in range(1, len(lat) - 1)] + [False])
        kg, h, sailed, end = sea.sail(lat, lon, on_link, kn, speeds_kn, at, at + every_hours, stops)
        seg_km = haversine_km(lat[:end], lon[:end], lat[1 : end + 1], lon[1 : end + 1])
        log.append({
            "time": str(now), "run": str(run), "from": [round(float(v), 4) for v in pos], "deadline": str(target),
            "planned_t": f["fuel_t"], "planned_h": f["hours"], "seconds": round(time.perf_counter() - t0, 1),
            "tries": sum(len(t) for t in p["tries"]),
            "sailed_km": round(float(seg_km.sum()), 1), "sailed_h": round(float(h.sum()), 1), "sailed_t": round(float(kg.sum()) / 1000, 2),
        })  # fmt: skip
        print(f"  {log[-1]}", flush=True)
        if not track["lat"]:
            track["lat"].append(float(lat[0]))
            track["lon"].append(float(lon[0]))
        track["lat"] += lat[1 : end + 1].tolist()
        track["lon"] += lon[1 : end + 1].tolist()
        track["on_link"] += on_link[:end].tolist()
        track["segment_kn"] += sailed.tolist()
        track["kg"] += kg.tolist()
        track["hours"] += h.tolist()
        at += float(h.sum())
        legs_km = np.cumsum([leg["km"] for leg in f["legs"]])[:-1]
        passed = int((legs_km <= seg_km.sum() + 0.01).sum())
        ahead, prices = ahead[passed:], p["prices"][passed:]  # the next plan starts from this one's prices
        if end == len(lat) - 1:
            break
        pos = (float(lat[end]), float(lon[end]))

    s = first["shortest"]
    s_lat, s_lon = np.array(s["lat"]), np.array(s["lon"])
    s_kg, s_h, _, _ = sea.sail(s_lat, s_lon, np.array(s["on_link"], bool), np.array(s["segment_kn"]), speeds_kn, 0.0)
    return {
        "departure": str(departure), "deadline": str(deadline), "hours": hours, "every_hours": every_hours,
        "planned": {"shortest_t": s["fuel_t"], "least_fuel_t": first["least_fuel"]["fuel_t"]},
        "shortest": summary(s_lat, s_lon, s_kg, s_h, hours),
        "sailed": summary(np.array(track["lat"]), np.array(track["lon"]), np.array(track["kg"]), np.array(track["hours"]), hours)
        | {"plans": len(log), "deadline_moved": late, "log": log, "on_link": track["on_link"], "segment_kn": track["segment_kn"]},
    }  # fmt: skip


def summary(lat, lon, kg, h, hours: float) -> dict:
    return {
        "km": round(float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()), 1),
        "hours": round(float(h.sum()), 1),
        "late_h": round(max(float(h.sum()) - hours, 0.0), 1),
        "fuel_t": round(float(kg.sum()) / 1000, 2),
        "lat": np.round(lat, 4).tolist(),
        "lon": np.round(lon, 4).tolist(),
    }


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
    print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "on_link", "segment_kn", "log")} if isinstance(v, dict) else v)
           for k, v in out.items()}, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
