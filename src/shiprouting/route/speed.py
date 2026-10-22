"""First attempt at choosing the speed: each move at whichever of a few speeds burns the least fuel.

Every speed has its own FuelCost in moving weather. When the search expands a cell, each
move is costed at every speed, at the time the ship reaches the cell, and the cheapest is
taken; the clock (hours) advances by that move's time at that speed. Links are taken at
the cheapest calm-water speed. There is no arrival time to meet.
"""

import time
from dataclasses import dataclass

import numpy as np

from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph


@dataclass
class SpeedChoice:
    costs: list[FuelCost]  # one per speed, moving weather, same departure

    def edge_cost(self, graph: Graph):
        """edge_cost function for quadtree.astar, its clock in hours."""
        per_speed = [c.costs(graph) for c in self.costs]

        def costs(c, hours):
            kg, h = (np.array(v) for v in zip(*(f(c, hours) for f in per_speed)))  # (speeds, moves)
            pick = np.argmin(kg, axis=0)
            moves = np.arange(kg.shape[1])
            return kg[pick, moves], h[pick, moves]

        return costs

    def link_cost(self, link, km: float, hours: float) -> tuple[float, float]:
        return min((c.link_cost(link, km, hours) for c in self.costs), key=lambda kg_h: kg_h[0])

    def speeds_kn(self, r: SearchResult) -> np.ndarray:
        """The speed of each segment of a path found with edge_cost, knots."""
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        return seg_km / np.diff(r.clock) / 1.852

    def path(self, r: SearchResult) -> float:
        """Fuel, kg, along a path found with edge_cost, each segment at its own speed and time."""
        kn = self.speeds_kn(r)
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        kg = 0.0
        for c in self.costs:
            mine = np.isclose(kn, c.speed / KNOT, rtol=1e-6)
            kg += seg_km[mine & r.on_link].sum() * c.calm_kg_per_km
            a = np.nonzero(mine & ~r.on_link)[0]
            kg += c.kg(r.lat[a], r.lon[a], r.lat[a + 1], r.lon[a + 1], r.clock[a]).sum()
        return float(kg)


def run(graph: Graph, ship, forecast, speeds_kn, fixed_kn: float, max_seconds: float) -> list[dict]:
    """For each test voyage, the moving-weather route at a fixed speed against the one choosing speeds."""
    from shiprouting.fuel.waves import AddedResistanceTable
    from shiprouting.sea.latlon import ROUTES
    from shiprouting.sea.quadtree import astar

    departure = forecast.time[0]
    costs = {}
    for kn in sorted({*speeds_kn, fixed_kn}):
        t0 = time.perf_counter()
        waves = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        costs[kn] = FuelCost(ship, forecast, departure, kn * KNOT, waves, frozen=False)
        print(f"{kn:g} kn: wave added resistance table built in {time.perf_counter() - t0:.0f} s", flush=True)
    fixed, choice = costs[fixed_kn], SpeedChoice([costs[kn] for kn in speeds_kn])
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "departure": str(departure)}
        for key, cost, link_cost in (("fixed", fixed.costs(graph), fixed.link_cost), ("choice", choice.edge_cost(graph), choice.link_cost)):
            try:
                r = astar(graph, s, g, cost, link_cost, lambda n: 0.0, max_seconds)
            except TimeoutError as e:
                row[key] = {"error": str(e)}
                continue
            kn = np.full(len(r.lat) - 1, fixed_kn) if key == "fixed" else choice.speeds_kn(r)
            row[key] = {
                "km": round(r.km),
                "hours": round(float(r.clock[-1]), 1),
                "fuel_t": round(r.cost / 1000, 2),
                "speeds_kn": {f"{v:g}": int(n) for v, n in zip(*np.unique(np.round(kn, 3), return_counts=True))},
                "expanded": r.expanded,
                "seconds": round(r.seconds, 1),
                "links": r.links,
                "lat": np.round(r.lat, 4).tolist(),
                "lon": np.round(r.lon, 4).tolist(),
                "clock": np.round(r.clock, 2).tolist(),
            }
        print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "clock")} if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Fuel-optimal routes choosing the speed of each move, in moving weather.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--speeds-kn", default="12,14,16,18,20", help="speeds to choose from, comma-separated")
    parser.add_argument("--fixed-kn", type=float, default=16.0, help="the fixed speed to compare with")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
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
    rows = run(graph, KCS_SHIP, forecast, [float(v) for v in args.speeds_kn.split(",")], args.fixed_kn, args.max_seconds)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
