"""How routes planned on a forecast fare in the weather that actually came.

A forecast run covers 15 days (360 h); longer voyages sail past its end, where
FuelCost holds the forecast's last step. To see what that costs, a voyage is
planned on one run and then costed in the weather that came, pieced together
from the first hours (steps 0–21 h) of each later daily run, the closest
public stand-in for observations (ecmwf.main --days).

Each voyage gets three routes at one fixed speed in moving weather: the
shortest, the one planned on the forecast, and the one that would have been
planned knowing the weather that came (the best possible). Each is costed in
both the forecast and the weather that came.
"""

import time

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import Graph, astar
from shiprouting.weather.forecast import Forecast


def run(graph: Graph, ship: Ship, forecast: Forecast, came: Forecast, speed_kn: float, max_seconds: float,
        tables: str | None = None) -> list[dict]:
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
    rows = run(graph, KCS_SHIP, forecast, came, args.speed_kn, args.max_seconds, args.tables)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
