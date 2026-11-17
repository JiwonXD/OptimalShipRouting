"""Check the planner against Zermelo's navigation problem, which has a known answer.

A ship at a constant speed V through the water crosses a current that runs east and grows with
the distance north, u = s * y. The least-time track bends north into the current. Its heading
obeys tan(theta) = tan(theta0) - s t (Zermelo), so the track is symmetric, the time is
T = 2 tan(theta0) / s, and theta0 is found from the distance to go,
x = V / s (ln(sec theta0 + tan theta0) + sec theta0 tan theta0) (east, at the equator, where
the sea is flat enough over a few hundred km: the error of taking it flat is (L/R)^2 ~ 1e-3
at most).

The planner is given that current, calm air and sea, one speed and a loose deadline. With one
speed and no wind or waves the fuel rate is the same everywhere, so least fuel is least time,
and its hours, sailed again in the evaluator, can be set against T.

    python -m shiprouting.route.zermelo [--roots 0.5,0.25,0.125,0.0625] [--shear 3e-5] [--km 400]
"""

import argparse
import dataclasses
import math
import time

import numpy as np
import shapely
from scipy.optimize import brentq

from shiprouting.fuel import waves
from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.route.evaluate import Sea
from shiprouting.route.fuel import KNOT
from shiprouting.route.plan import plan
from shiprouting.sea.land import Land
from shiprouting.sea.quadtree import Graph
from shiprouting.weather.forecast import Forecast

R_KM = 6371.0088  # mean earth radius, as haversine_km
T0 = np.datetime64("2026-01-01T00:00", "s")


def least_time(x_m: float, V: float, s: float) -> tuple[float, float]:
    """Least time, s, from (0, 0) to (x_m, 0) at speed V (m/s) in the current u = s y, and the
    farthest the track goes north, m."""
    if s == 0:
        return x_m / V, 0.0

    def x_of(theta0):  # east distance made from theta0 to -theta0
        sec, tan = 1 / math.cos(theta0), math.tan(theta0)
        return V / s * (math.log(sec + tan) + sec * tan)

    theta0 = brentq(lambda th: x_of(th) - x_m, 1e-9, math.pi / 2 - 1e-6, xtol=1e-14)
    return 2 * math.tan(theta0) / s, V / s * (1 / math.cos(theta0) - 1)


def shear(lat0: float, s: float) -> Forecast:
    """Calm air and sea, a current east of s * (distance north of lat0), on a fine grid near the equator."""
    lat, lon = np.arange(-10.0, 10.001, 0.25), np.arange(0.0, 360.0, 0.25)
    u = s * np.radians(lat - lat0) * R_KM * 1000
    full = (2, len(lat), len(lon))
    fields = {n: np.zeros(full) for n in ("wind_u", "wind_v", "Hs", "wave_from_x", "wave_from_y", "T_m10", "current_v")}
    fields["T_m10"][:] = 8.0
    fields["current_u"] = np.broadcast_to(u[None, :, None], full).copy()
    return Forecast(lat, lon, np.array([T0, T0 + np.timedelta64(1000, "h")]), fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="The planner against Zermelo's least-time track in a shear current.")
    parser.add_argument("--roots", default="0.5,0.25,0.125,0.0625", help="grid cell sizes, degrees")
    parser.add_argument("--shear", type=float, default=3e-5, help="s, 1/s: the current grows by s m/s per m north")
    parser.add_argument("--km", type=float, default=400.0, help="distance to go east")
    parser.add_argument("--kn", type=float, default=16.0, help="speed through the water")
    args = parser.parse_args()

    # Wave tables are not needed in a calm sea: build a coarse one quickly.
    build = waves.AddedResistanceTable.build.__func__
    waves.AddedResistanceTable.build = classmethod(
        lambda cls, sk, speed, s=1.0, T01=None, angle=None: build(cls, sk, speed, s, (7.0, 9.0), (0.0, math.pi / 2, math.pi)))
    ship = dataclasses.replace(KCS_SHIP, windage=None)  # no air resistance: the fuel rate is the same everywhere
    land = Land([shapely.box(100.0, 30.0, 101.0, 31.0)])  # far away; given so the plan straightens and smooths
    V = args.kn * KNOT
    print(f"V {args.kn:g} kn, shear {args.shear:g} 1/s, {args.km:g} km east")
    print("root°  cells   start            goal             exact_h  plan_h   error%  north_km(exact/plan)  s")
    for root in (float(v) for v in args.roots.split(",")):
        graph = Graph.build(land, root=root, min_step=root, lat_limits=(-6.0, 6.0))
        a = graph.nearest(0.0, 150.0)
        dlon = args.km / (R_KM * math.cos(math.radians(graph.lat[a]))) * 180 / math.pi
        b = graph.nearest(float(graph.lat[a]), 150.0 + dlon)
        start, goal = (float(graph.lat[a]), float(graph.lon[a])), (float(graph.lat[b]), float(graph.lon[b]))
        x_m = math.radians(goal[1] - start[1]) * R_KM * 1000 * math.cos(math.radians(start[0]))
        T, north = least_time(x_m, V, args.shear)
        fc = shear(start[0], args.shear)
        t0 = time.perf_counter()
        out = plan(graph, ship, fc, start, goal, T0, T0 + np.timedelta64(int(3 * T), "s"), (args.kn,), land=land, keep_shortest=False)
        f = out["least_fuel"]
        _, h, _, _ = Sea(ship, fc, T0).sail(np.array(f["lat"]), np.array(f["lon"]), np.array(f["on_link"], bool),
                                            np.array(f["segment_kn"]), (args.kn,), 0.0)  # its hours unrounded, judged on its own
        hours = float(h.sum())
        plan_north = (max(f["lat"]) - start[0]) * math.pi / 180 * R_KM
        err = 100 * (hours * 3600 - T) / T
        print(f"{root:<6g} {len(graph.lat):<7d} {start[0]:7.4f},{start[1]:8.4f} {goal[0]:7.4f},{goal[1]:8.4f}"
              f" {T / 3600:8.3f} {hours:8.3f} {err:+7.3f}  {north / 1000:7.1f} / {plan_north:7.1f}"
              f"  {time.perf_counter() - t0:5.1f}", flush=True)


if __name__ == "__main__":
    main()
