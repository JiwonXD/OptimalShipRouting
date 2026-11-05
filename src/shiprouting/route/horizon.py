"""How routes planned on a forecast fare in the weather that actually came.

A forecast run covers 15 days (360 h); longer voyages sail past its end, where
FuelCost holds the forecast's last step. To see what that costs, a voyage is
planned on one run and then costed in the weather that came, pieced together
from the first hours (steps 0–21 h) of each later daily run, the closest
public stand-in for observations (ecmwf.main --days).

Each voyage gets three routes at one fixed speed in moving weather: the
shortest, the one planned on the forecast, and the one that would have been
planned knowing the weather that came (the best possible). Each is costed in
both the forecast and the weather that came. Given the later runs, a fourth is
replanned as a ship would: every few days, from where it has got to, on the
run issued that day (replan).
"""

import time

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph, astar
from shiprouting.weather.forecast import Forecast


def run(graph: Graph, ship: Ship, forecast: Forecast, came: Forecast, speed_kn: float, max_seconds: float,
        tables: str | None = None, runs=None, every_days: int = 5) -> list[dict]:
    """runs(time) gives the forecast run issued at a time, or None; without it there is no replanning."""
    from shiprouting.sea.latlon import ROUTES

    departure = forecast.time[0]
    speed = speed_kn * KNOT
    if tables:
        waves = AddedResistanceTable.cached(tables, ship.seakeeping, speed)
    else:
        waves = AddedResistanceTable.build(ship.seakeeping, speed)
    planned = FuelCost(ship, forecast, departure, speed, waves, frozen=False)
    actual = FuelCost(ship, came, departure, speed, waves, frozen=False)
    forecast_hours = float((forecast.time[-1] - departure) / np.timedelta64(1, "h"))
    came_hours = float((came.time[-1] - departure) / np.timedelta64(1, "h"))
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "speed_kn": speed_kn, "departure": str(departure), "forecast_hours": forecast_hours, "came_hours": came_hours}
        found = {"shortest": astar(graph, s, g)}
        for key, cost in (("planned", planned), ("best", actual)):
            t0 = time.perf_counter()
            found[key] = astar(graph, s, g, cost.costs(graph), cost.link_cost, lambda n: 0.0, max_seconds)
            print(f"  {key}: searched in {time.perf_counter() - t0:.0f} s", flush=True)
        if runs is not None:
            t0 = time.perf_counter()
            found["replanned"] = replan(graph, ship, waves, speed, departure, g, found["planned"], runs, every_days, max_seconds)
            print(f"  replanned: {time.perf_counter() - t0:.0f} s", flush=True)
        for key, r in found.items():
            seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
            hours = np.cumsum(seg_km) / (speed * 3.6)
            row[key] = {
                "km": round(r.km),
                "hours": round(float(hours[-1]), 1),
                "fuel_forecast_t": round(planned.path(r.lat, r.lon, r.on_link) / 1000, 2),
                "fuel_came_t": round(actual.path(r.lat, r.lon, r.on_link) / 1000, 2),
                "km_after_forecast": round(float(seg_km[hours > forecast_hours].sum())),
                "links": r.links,
            }
        print(row, flush=True)
        for key, r in found.items():
            row[key] |= {"lat": np.round(r.lat, 4).tolist(), "lon": np.round(r.lon, 4).tolist()}
        rows.append(row)
    return rows


def replan(graph: Graph, ship: Ship, waves, speed: float, departure, g: int, plan: SearchResult, runs, every_days: int,
           max_seconds: float) -> SearchResult:
    """The way sailed when, every every_days, the ship plans again from where it is on the run issued then.

    It replans from the first cell of the current plan it reaches at or after that day (a cell,
    not a point inside a link).
    """
    lat, lon, on_link, links = [], [], [], []
    at, day = 0.0, every_days  # hours since departure at the start of `plan`; the next day to replan
    while True:
        seg_km = haversine_km(plan.lat[:-1], plan.lon[:-1], plan.lat[1:], plan.lon[1:])
        hours = at + np.r_[0.0, np.cumsum(seg_km)] / (speed * 3.6)
        cell = ~np.r_[False, plan.on_link] & ~np.r_[plan.on_link, False]  # neither segment next to it is on a link
        cell[[0, -1]] = False
        fc = None
        while day * 24 < hours[-1] and fc is None:  # the first day a run exists, before the plan ends
            fc = runs(departure + np.timedelta64(day * 24, "h"))
            day += every_days if fc is None else 0
        cut = np.nonzero(cell & (hours >= day * 24))[0] if fc is not None else []
        end = int(cut[0]) if len(cut) else len(plan.lat) - 1
        lat += list(plan.lat[: end + 1]) if not lat else list(plan.lat[1 : end + 1])
        lon += list(plan.lon[: end + 1]) if not lon else list(plan.lon[1 : end + 1])
        on_link += list(plan.on_link[:end])
        links += [k for k in plan.links if k not in links]
        if not len(cut):
            break
        at = float(hours[end])
        s = graph.nearest(plan.lat[end], plan.lon[end])
        cost = FuelCost(ship, fc, departure + np.timedelta64(round(at * 3600), "s"), speed, waves, frozen=False)
        plan = astar(graph, s, g, cost.costs(graph), cost.link_cost, lambda n: 0.0, max_seconds)
        print(f"    day {day}: replanned from ({plan.lat[0]:.2f}, {plan.lon[0]:.2f}) at {at:.0f} h", flush=True)
        day += every_days
    lat, lon = np.array(lat), np.array(lon)
    km = float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
    return SearchResult(lat, lon, np.array(on_link, bool), links, km, 0, 0.0)


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Routes planned on a forecast, costed in the weather that came.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the GRIB2 files of the run planned on")
    parser.add_argument("came", help="glob of the GRIB2 files of the weather that came (ecmwf --days)")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--speed-kn", type=float, default=16.0)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
    parser.add_argument("--runs", help="glob of a later run's GRIB2 files with {run} for its YYYYMMDDHH, to replan on")
    parser.add_argument("--replan-days", type=int, default=5, help="replan every this many days")
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
    came = ecmwf.load(sorted(glob.glob(args.came)))
    runs = None
    if args.runs:
        def runs(t):
            files = sorted(glob.glob(args.runs.format(run=str(t.astype("datetime64[h]")).replace("-", "").replace("T", ""))))
            return ecmwf.load(files) if files else None

    rows = run(graph, KCS_SHIP, forecast, came, args.speed_kn, args.max_seconds, args.tables, runs, args.replan_days)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
