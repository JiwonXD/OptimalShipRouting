"""How much the surface current changes a route sailed at given speeds through the water.

The routes are searched with the current left out. Here a found route is sailed again
with it: the ship keeps each segment's speed through the water V and steers to stay on
its track, so the current across the track c_x takes part of V and the current along it
c_a adds to what is left. Over the ground it makes sqrt(V² − c_x²) + c_a. The fuel per
hour is that of V, so each segment's fuel scales with its hours. Wind and waves are
taken as without the current (the small crab angle is ignored), and links have no current.
The current is that at the segment's midpoint, at the time the ship gets there, held at
the forecast's last step beyond its end.
"""

import numpy as np

from shiprouting.route.fuel import KNOT, FuelCost, bearing
from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import along
from shiprouting.weather.forecast import Forecast


def over_ground(forecast: Forecast, lat, lon, kn, on_link, departure) -> tuple[np.ndarray, np.ndarray]:
    """Per segment: speed over the ground, knots, and the current along the track, knots,
    sailing the segments one after another from departure."""
    lat, lon, kn = np.asarray(lat, float), np.asarray(lon, float), np.asarray(kn, float)
    seg_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    mid_lat, mid_lon = along(lat[:-1], lon[:-1], lat[1:], lon[1:], 0.5)
    course = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    end = forecast.time[-1].astype("datetime64[s]")
    sog, c_along = kn.copy(), np.zeros(len(kn))
    hours = 0.0
    for k in range(len(seg_km)):
        if not on_link[k]:
            t = min(np.datetime64(departure, "s") + np.timedelta64(int((hours + seg_km[k] / 2 / (kn[k] * 1.852)) * 3600), "s"), end)
            u, v = (float(np.nan_to_num(forecast.sample(n, mid_lat[k], mid_lon[k], t))) / KNOT for n in ("current_u", "current_v"))
            c_along[k] = u * np.sin(course[k]) + v * np.cos(course[k])
            c_across = u * np.cos(course[k]) - v * np.sin(course[k])
            sog[k] = np.sqrt(max(kn[k] ** 2 - c_across**2, 0.0)) + c_along[k]
        hours += seg_km[k] / (max(sog[k], 1e-6) * 1.852)
    return sog, c_along


def effect(forecast: Forecast, ship, found: dict, departure, tables: str | None = None) -> dict:
    """A plan's route (lat, lon, on_link, segment_kn) sailed without and with the current."""
    from shiprouting.fuel.waves import AddedResistanceTable

    lat, lon = np.asarray(found["lat"], float), np.asarray(found["lon"], float)
    on_link, kn = np.asarray(found["on_link"], bool), np.asarray(found["segment_kn"], float)
    seg_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    sog, c_along = over_ground(forecast, lat, lon, kn, on_link, departure)
    still_h, current_h = seg_km / (kn * 1.852), seg_km / (np.maximum(sog, 1e-6) * 1.852)

    def fuel(seg_h):
        clock = np.r_[0.0, np.cumsum(seg_h)][:-1]
        kg = np.zeros(len(kn))
        for v in np.unique(kn):
            waves = (AddedResistanceTable.cached(tables, ship.seakeeping, v * KNOT) if tables
                     else AddedResistanceTable.build(ship.seakeeping, v * KNOT))
            cost = FuelCost(ship, forecast, np.datetime64(departure, "s"), v * KNOT, waves, frozen=False)
            mine = kn == v
            sea = np.nonzero(mine & ~on_link)[0]
            kg[sea] = cost.kg(lat[sea], lon[sea], lat[sea + 1], lon[sea + 1], clock[sea])
            kg[mine & on_link] = seg_km[mine & on_link] * cost.calm_kg_per_km
        return kg * seg_h / (seg_km / (kn * 1.852))  # kg is for sailing at V with no current

    still_t, current_t = fuel(still_h).sum() / 1000, fuel(current_h).sum() / 1000
    sea_h = np.where(on_link, 0.0, current_h)
    mid_lat, mid_lon = along(lat[:-1], lon[:-1], lat[1:], lon[1:], 0.5)
    return {
        "hours": round(float(still_h.sum()), 1),
        "hours_current": round(float(current_h.sum()), 1),
        "fuel_t": round(float(still_t), 2),
        "fuel_current_t": round(float(current_t), 2),
        "mean_along_kn": round(float((c_along * sea_h).sum() / sea_h.sum()), 2),
        "hours_against_1kn": round(float(sea_h[c_along <= -1].sum()), 1),
        "hours_with_1kn": round(float(sea_h[c_along >= 1].sum()), 1),
        "most_against": [(round(float(mid_lat[k]), 2), round(float(mid_lon[k]), 2), round(float(c_along[k]), 2))
                         for k in np.argsort(c_along)[:3]],
    }


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Plans' routes sailed again with the surface current.")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("plans", nargs="+", help="plan .json files (route.plan)")
    parser.add_argument("--tables", help="folder the wave added resistance tables are kept in")
    args = parser.parse_args()

    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    for path in args.plans:
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
        print(path, flush=True)
        for key in ("shortest", "least_fuel"):
            if key in out:
                print(f"  {key}: {effect(forecast, KCS_SHIP, out[key], out['departure'], args.tables)}", flush=True)


if __name__ == "__main__":
    main()
