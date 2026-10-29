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
for the voyage as a whole (legs="shared"). Given the land, the cells' steps are cut out:
the shortest route's legs are straightened to great-circle arcs (quadtree.straighten),
and the least-fuel route takes the shortcuts that burn no more fuel (SpeedChoice.shortcut).
Both routes then have their sharp turns smoothed out (quadtree.smooth_turns), waypoints
kept, and the least-fuel route is re-timed to the speeds on offer and spends the time
that saves.

With currents, the surface current changes each move's speed over the ground (FuelCost),
so the shortest route's one speed through the water is adjusted until it arrives at the
deadline.
"""

import math
import time

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.speed import SpeedChoice, meet_eta
from shiprouting.sea.latlon import ICE_FREE, SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph, astar, join_legs, smooth_turns, straighten
from shiprouting.weather.forecast import Forecast

SPEEDS_KN = (12, 14, 16, 18, 20, 22, 24)
MAX_MOVE_KM = 50.0  # a point farther than this from the nearest sea cell is taken to be a mistake


def plan(graph: Graph, ship: Ship, forecast: Forecast, start, goal, departure, deadline, speeds_kn=SPEEDS_KN,
         tables: str | None = None, max_tries: int = 12, max_seconds: float = math.inf, via=(), legs: str = "distance",
         land=None, currents: bool = True, prices=None, least_fuel: bool = True) -> dict:
    """least_fuel=False finds the shortest route only. prices: the price of an hour each leg met
    its time at in a like plan before (out["prices"] of the last plan of the same voyage, say),
    to start from; one for legs="shared"."""
    departure, deadline = np.datetime64(departure, "s"), np.datetime64(deadline, "s")
    hours = float((deadline - departure) / np.timedelta64(1, "h"))
    points = [start, *via, goal]
    names = ["출발지", *(f"경유지 {k}" for k in range(1, len(via) + 1)), "도착지"]
    out = {
        "start": list(start), "goal": list(goal), "via": [list(p) for p in via], "legs_time": legs, "currents": currents,
        "departure": str(departure), "deadline": str(deadline), "hours": hours,
    }
    if not forecast.time[0] <= departure < forecast.time[-1]:
        out["error"] = f"출발 시각 {departure}이 예보 범위({forecast.time[0]} ~ {forecast.time[-1]}) 밖이에요."
        return out
    if hours <= 0:
        out["error"] = "도착 기한은 출발 시각보다 뒤여야 해요."
        return out
    for name, p in zip(names, points):
        if not -90 <= p[0] <= 90:
            out["error"] = f"{name}의 위도 {p[0]:g}는 -90 ~ 90 사이여야 해요."
            return out
    nodes = [graph.nearest(*p) for p in points]
    out["moved_km"] = [round(float(haversine_km(graph.lat[n], graph.lon[n], *p)), 1) for n, p in zip(nodes, points)]
    for name, km in zip(names, out["moved_km"]):
        if km > MAX_MOVE_KM:
            out["error"] = (f"{name}에서 가장 가까운 바다 칸이 {km:,.0f} km 떨어져 있어요. 바다나 항구를 찍어 주세요"
                            f" (바다 지도는 남위 {-ICE_FREE[0]:g}° ~ 북위 {ICE_FREE[1]:g}°만 다뤄요).")
            return out
    if len(set(nodes)) == 1:
        out["error"] = "출발지와 도착지가 같은 바다 칸이에요."
        return out

    waves = {}

    def cost(kn, at=departure):
        if kn not in waves:
            if tables:
                waves[kn] = AddedResistanceTable.between(tables, ship.seakeeping, kn * KNOT)
            else:
                waves[kn] = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        return FuelCost(ship, forecast, at, kn * KNOT, waves[kn], frozen=False, currents=currents)

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
    shortest = []
    for k, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        try:
            shortest.append(astar(graph, a, b))
        except ValueError:  # searched all the water a reaches
            out["error"] = f"{names[k]}와 {names[k + 1]} 사이에 이어진 바닷길이 없어요 (카스피해처럼 바다로 이어지지 않은 물일 수 있어요)."
            return out
    if land is not None:  # the cells' path steps between grid directions: cut it to great-circle arcs
        shortest = [straighten(land, leg, piece_km=20.0) for leg in shortest]
    r = smoothed(join_legs(shortest))
    need_kn = r.km / hours / 1.852
    if need_kn > max(speeds_kn):
        out["error"] = (f"도착 기한을 맞출 수 없어요: 최단 경로({r.km:,.0f} km)로도 평균 {need_kn:.1f}노트가 필요한데"
                        f" 최고 속력은 {max(speeds_kn):g}노트예요.")
        return out
    kn = max(need_kn, min(speeds_kn))
    for _ in range(6):  # the current makes it arrive early or late: adjust the speed to arrive just before the deadline
        fuel, clock = cost(kn).sail(r.lat, r.lon, r.on_link)
        if hours * (1 - 0.002) <= clock[-1] <= hours or (kn == min(speeds_kn) and clock[-1] < hours):
            break
        kn = max(round(kn * clock[-1] / (hours * (1 - 0.001)), 2), min(speeds_kn))
    if kn > max(speeds_kn):
        out["error"] = (f"도착 기한을 맞출 수 없어요: 해류까지 치면 최단 경로({r.km:,.0f} km)로도 {kn:.1f}노트가 필요한데"
                        f" 최고 속력은 {max(speeds_kn):g}노트예요.")
        return out
    out["shortest"] = describe(r, fuel, np.full(len(r.lat) - 1, kn), float(clock[-1]), time.perf_counter() - t0)
    if not least_fuel:
        return out

    t0 = time.perf_counter()
    # The shortest route's own speed is offered too, so sailing it as it does is among the choices.
    offered = sorted({*speeds_kn, kn})
    choice = SpeedChoice([cost(k) for k in offered])
    prices = list(prices or [])
    if legs == "shared":
        f, tries = meet_eta(graph, nodes[0], nodes[-1], choice, hours, max_tries=max_tries, max_seconds=max_seconds,
                            via=nodes[1:-1], price=(prices or [None])[0])
        out["tries"] = [tries]
    else:
        found, out["tries"], at = [], [], departure
        for i, ((a, b), leg) in enumerate(zip(zip(nodes[:-1], nodes[1:]), shortest)):
            if a == b:  # a waypoint on the point before it: nothing to sail
                found.append(leg)
                out["tries"].append([])
                continue
            f, tries = meet_eta(graph, a, b, SpeedChoice([cost(k, at) for k in offered]), hours * leg.km / sum(r.km for r in shortest),
                                max_tries=max_tries, max_seconds=max_seconds, price=prices[i] if i < len(prices) else None)
            out["tries"].append(tries)
            if f is None:
                break
            found.append(f)
            at = at + np.timedelta64(int(round(f.clock[-1] * 3600)), "s")
        f = join_legs(found) if f is not None else None
    if f is None:
        out["error"] = "속력을 골라 가는 경로 중 도착 기한 안에 닿는 것을 찾지 못했어요."
        return out
    out["prices"] = [min((t["per_hour"] for t in tries if t["in_time"]), default=None) for tries in out["tries"]]  # the lowest that arrived in time
    if land is not None:
        f = choice.fill_slack(choice.snap_up(smoothed(choice.shortcut(land, f, waypoints(f)))), hours)
    out["least_fuel"] = describe(f, f.cost, choice.speeds_kn(f), float(f.clock[-1] - f.clock[0]), time.perf_counter() - t0)
    ends = [0, *waypoints(f), len(f.lat) - 1]
    seg_km = np.r_[0.0, np.cumsum(haversine_km(f.lat[:-1], f.lon[:-1], f.lat[1:], f.lon[1:]))]
    out["least_fuel"]["legs"] = [{"km": round(float(seg_km[b] - seg_km[a]), 1), "hours": round(float(f.clock[b] - f.clock[a]), 1)}
                                 for a, b in zip(ends[:-1], ends[1:])]
    saving = out["shortest"]["fuel_t"] - out["least_fuel"]["fuel_t"]
    out["saving_t"] = round(saving, 2)
    out["saving_pct"] = round(100 * saving / out["shortest"]["fuel_t"], 2)
    return out


def describe(r: SearchResult, fuel_kg: float, speeds_kn, hours: float, seconds: float) -> dict:
    return {
        "km": round(r.km, 1),
        "hours": round(hours, 1),
        "fuel_t": round(fuel_kg / 1000, 2),
        "speeds_kn": {f"{v:g}": int(n) for v, n in zip(*np.unique(np.round(speeds_kn, 2), return_counts=True))},
        "links": r.links,
        "seconds": round(seconds, 1),
        "lat": np.round(r.lat, 4).tolist(),
        "lon": np.round(r.lon, 4).tolist(),
        "on_link": r.on_link.tolist(),
        "segment_kn": np.round(speeds_kn, 2).tolist(),
    }


def add_sea_args(parser) -> None:
    """The command-line options for the sea graph (see sea_graph)."""
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--bathymetry", help="GEBCO grid, netCDF: keep out of water --min-depth m deep or less")
    parser.add_argument("--min-depth", type=float, default=15.0)
    parser.add_argument("--link-approach-km", type=float, default=10.0, help="no depth check this close to a link's sea end")
    parser.add_argument("--graph", help=".npz file to keep the graph in between runs; rebuilt when the options above change")


def sea_graph(args):
    """The land (with the shallow water, given the bathymetry) and the sea graph, from add_sea_args' options."""
    import json

    from shiprouting.sea.land import load_osm_land_polygons

    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    channels = {}
    if args.bathymetry:
        from shiprouting.sea.depth import CHANNELS, Bathymetry, LandOrShallow, link_ends

        bathy = Bathymetry(args.bathymetry)
        channels = CHANNELS
        land = LandOrShallow(land, bathy, args.min_depth, link_ends(), args.link_approach_km)
    t0 = time.perf_counter()
    key = json.dumps({k: getattr(args, k) for k in ("land", "min_step", "links", "bathymetry", "min_depth", "link_approach_km")}
                     | ({"channels": channels} if channels else {}))

    def build():
        from shiprouting.sea.depth import channel

        traced = [channel(bathy, land.land, name, a, b, args.min_depth) for name, (a, b) in channels.items()]
        return Graph.build(land, min_step=args.min_step, links=links + traced)

    graph = Graph.cached(args.graph, key, build)
    print(f"graph: {len(graph.lat)} cells, links {[k.name for *_, k in graph.links]}, ready in {time.perf_counter() - t0:.0f} s", flush=True)
    return land, graph


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
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
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
    parser.add_argument("--max-tries", type=int, default=12, help="searches to meet the deadline")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    add_sea_args(parser)
    parser.add_argument("--legs", choices=["distance", "shared"], default="distance",
                        help="time per leg: in proportion to distance, or one price of an hour for the whole voyage")
    parser.add_argument("--no-smoothing", action="store_true", help="leave sharp turns as found")
    parser.add_argument("--no-currents", action="store_true", help="leave the surface current out")
    args = parser.parse_args()

    land, graph = sea_graph(args)
    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    start, goal, *via = [tuple(float(v) for v in p.split(",")) for p in (args.start, args.goal, *args.via)]
    speeds = [float(v) for v in args.speeds_kn.split(",")]
    out = plan(graph, KCS_SHIP, forecast, start, goal, args.departure, args.deadline, speeds, args.tables, args.max_tries,
               args.max_seconds, via, args.legs, None if args.no_smoothing else land, not args.no_currents)
    print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "on_link", "segment_kn", "clock")} if isinstance(v, dict) else v)
           for k, v in out.items() if k != "tries"}, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
