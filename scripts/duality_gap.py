"""How far the price-of-an-hour search (meet_eta) is from the least fuel that meets the deadline.

Both search the same graph with the same speeds and deadline; route.exact.least_fuel_by keeps
time in the state and is exact but for buckets of 0.1 h. The gap is meet_eta's fuel over the
exact fuel, minus one, on the grid (before shortcuts and smoothing).

    python scripts/duality_gap.py synthetic [--cases 14] [--dt-hours 0.1] [--max-labels N]
    python scripts/duality_gap.py real LAND --runs DIR [--graph ...] [--tables DIR] [--cases 12] [--dt-hours 0.1] [--max-labels N]

A case whose exact search runs past max_labels is given up and counted as such. With
--corridor-km the exact search keeps within that distance of meet_eta's path, so the gap is
the one within that corridor.

synthetic: a calm-sea graph near the equator and a storm of random size, strength, place and
drift; deadlines of 19-26 h for about 670 km.
real: open-ocean stretches of 700-900 km (North Pacific, North Atlantic, Indian Ocean) on the
real sea graph, leaving at 00z on the forecast runs found, deadlines at 15-17 kn.
"""

import argparse
import dataclasses
import math
import time

import numpy as np

from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.exact import least_fuel_by
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.speed import SpeedChoice, meet_eta
from shiprouting.sea.latlon import haversine_km
from shiprouting.weather.forecast import Forecast

SPEEDS = (12, 14, 16, 18, 20, 22, 24)
T0 = np.datetime64("2026-01-01T00:00", "s")
STRETCHES = {  # open ocean, no land or links between
    "north_pacific": ((38.0, 160.0), (40.0, 169.0)),
    "north_atlantic": ((46.0, -40.0), (49.0, -30.0)),
    "indian": ((-8.0, 70.0), (-4.0, 77.0)),
}


def storm(lat0, lon0, dlon_per_h, peak_hs, radius) -> Forecast:
    lat, lon = np.arange(-20, 20.01, 0.5), np.arange(0, 360, 0.5)
    times = T0 + np.arange(0, 61, 3) * np.timedelta64(3600, "s")
    L, O = np.meshgrid(lat, lon, indexing="ij")
    names = ("wind_u", "wind_v", "Hs", "wave_from_x", "wave_from_y", "T_m10", "current_u", "current_v")
    f = {n: np.zeros((len(times), len(lat), len(lon))) for n in names}
    for k in range(len(times)):
        w = np.exp(-((L - lat0) ** 2 + (O - lon0 - dlon_per_h * 3 * k) ** 2) / radius**2)
        f["Hs"][k] = 1.0 + peak_hs * w
        f["T_m10"][k] = 7.0 + 3 * w
        f["wave_from_x"][k] = -1.0  # head seas going east
        f["wind_u"][k] = -(4 + 16 * w)
    return Forecast(lat, lon, times, f)


def corridor(graph, r, km: float) -> np.ndarray | None:
    """The nodes within km of path r (None: no limit)."""
    if not km:
        return None
    near = np.zeros(len(graph.lat), bool)
    box = ((graph.lat > r.lat.min() - km / 100) & (graph.lat < r.lat.max() + km / 100)).nonzero()[0]
    for la, lo in zip(r.lat, r.lon):
        near[box] |= haversine_km(graph.lat[box], graph.lon[box], la, lo) <= km
    return near


def gap(graph, a, b, choice, eta, dt_hours=0.1, max_labels=5_000_000, corridor_km=0.0) -> dict:
    r, _ = meet_eta(graph, a, b, choice, eta)
    lag = choice.path(r)
    t0 = time.perf_counter()
    try:  # meet_eta's path bounds the fuel from the start
        ex = least_fuel_by(graph, a, b, choice, eta, dt_hours, max_labels=max_labels, upper_kg=lag * (1 + 1e-6),
                           allowed=corridor(graph, r, corridor_km))
    except TimeoutError as e:
        return {"eta_h": round(eta, 2), "price_t": round(lag / 1000, 3), "price_h": round(float(r.clock[-1]), 2),
                "failed": str(e), "exact_s": round(time.perf_counter() - t0, 1)}
    except ValueError:  # nothing in time burns less than meet_eta's path (as the buckets see it): no gap
        ex = dataclasses.replace(r, cost=lag, expanded=-1)
    same = len(ex.lat) == len(r.lat) and bool(np.allclose(ex.lat, r.lat) and np.allclose(ex.lon, r.lon))
    return {"eta_h": round(eta, 2), "price_t": round(lag / 1000, 3), "price_h": round(float(r.clock[-1]), 2),
            "exact_t": round(ex.cost / 1000, 3), "exact_h": round(float(ex.clock[-1]), 2), "same_path": same,
            "labels": ex.expanded, "exact_s": round(time.perf_counter() - t0, 1), "gap_pct": round(100 * (lag - ex.cost) / ex.cost, 3)}


def synthetic(args) -> list[dict]:
    import shapely

    from shiprouting.fuel import waves
    from shiprouting.sea.land import Land
    from shiprouting.sea.quadtree import Graph

    build = waves.AddedResistanceTable.build.__func__
    waves.AddedResistanceTable.build = classmethod(lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(
        cls, sk, speed, s, (6.0, 8.0, 10.0), tuple(np.linspace(0, math.pi, 5))))
    graph = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=0.5, min_step=0.5, lat_limits=(-8.0, 8.0))
    a, b = graph.nearest(0.25, 150.25), graph.nearest(0.25, 156.25)
    rng = np.random.default_rng(1)
    out = []
    for case in range(args.cases):
        fc = storm(rng.uniform(-2, 2), rng.uniform(150, 156), rng.uniform(-0.2, 0.2), rng.uniform(3, 7), rng.uniform(1, 2.5))
        eta = rng.uniform(19, 26)
        choice = SpeedChoice([FuelCost(KCS_SHIP, fc, T0, k * KNOT, frozen=False, currents=True) for k in SPEEDS])
        out.append({"case": case} | gap(graph, a, b, choice, eta, args.dt_hours, args.max_labels, args.corridor_km))
        print(out[-1], flush=True)
    return out


def real(args) -> list[dict]:
    from shiprouting.fuel.waves import AddedResistanceTable
    from shiprouting.route.plan import sea_graph
    from shiprouting.route.voyage import Runs

    _, graph = sea_graph(args)
    runs = Runs(args.runs, lag_hours=0.0)
    rng = np.random.default_rng(2)
    out = []
    picks = rng.choice(len(runs.times), size=min(args.cases, len(runs.times)), replace=False)
    for case, k in enumerate(sorted(picks)):
        name = list(STRETCHES)[case % len(STRETCHES)]
        (la0, lo0), (la1, lo1) = STRETCHES[name]
        t, fc = runs.newest(runs.times[k])
        a, b = graph.nearest(la0, lo0), graph.nearest(la1, lo1)
        km = float(haversine_km(graph.lat[a], graph.lon[a], graph.lat[b], graph.lon[b]))
        eta = km / (rng.uniform(15, 17) * 1.852)
        costs = []
        for kn in SPEEDS:
            w = AddedResistanceTable.between(args.tables, KCS_SHIP.seakeeping, kn * KNOT) if args.tables else None
            costs.append(FuelCost(KCS_SHIP, fc, t, kn * KNOT, w, frozen=False, currents=True))
        out.append({"case": case, "stretch": name, "run": str(t)} | gap(graph, a, b, SpeedChoice(costs), eta, args.dt_hours, args.max_labels, args.corridor_km))
        print(out[-1], flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="meet_eta against the exact least fuel by a deadline.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--cases", type=int, default=14)
    p.add_argument("--dt-hours", type=float, default=0.1, help="time bucket of the exact search")
    p.add_argument("--max-labels", type=int, default=5_000_000, help="give a case up after this many labels")
    p.add_argument("--corridor-km", type=float, default=0.0, help="exact search only this close to meet_eta's path (0: anywhere)")
    p = sub.add_parser("real")
    p.add_argument("--dt-hours", type=float, default=0.1, help="time bucket of the exact search")
    p.add_argument("--max-labels", type=int, default=5_000_000, help="give a case up after this many labels")
    p.add_argument("--corridor-km", type=float, default=0.0, help="exact search only this close to meet_eta's path (0: anywhere)")
    p.add_argument("land", help="land polygons shapefile")
    p.add_argument("--runs", required=True)
    p.add_argument("--tables")
    p.add_argument("--cases", type=int, default=12)
    from shiprouting.route.plan import add_sea_args

    add_sea_args(p)
    args = parser.parse_args()
    rows = synthetic(args) if args.cmd == "synthetic" else real(args)
    done = [r for r in rows if "gap_pct" in r]
    gaps = np.array([r["gap_pct"] for r in done])
    print(f"{len(rows)} cases, {len(rows) - len(done)} given up: median gap {np.median(gaps):.3f}%, largest {gaps.max():.3f}%,"
          f" over 0.5%: {(gaps > 0.5).sum()}, same path in {sum(r['same_path'] for r in done)}" if len(done) else f"{len(rows)} cases, all given up")


if __name__ == "__main__":
    main()
