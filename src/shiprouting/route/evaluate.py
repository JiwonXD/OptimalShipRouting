"""Sail a track in given weather and count its fuel and hours: the one place every voyage is judged.

A track is its points, which segments are links, and the speed through the water planned for
each segment. Sea sails it from a clock (hours since departure) in a Forecast, which can be any
weather on the Forecast grid: the forecast a plan was made on, the weather that came, or another
source. Where a planned speed needs more than the engine's MCR, the segment is sailed at the
fastest slower speed it can make, of the planned ones and those offered.

Planning and judging are kept apart so that a plan can be judged in weather it was not made
on. voyage.py sails its plans here; recost() sails the tracks of a saved voyage again in other
weather.
"""

import dataclasses
import math

import numpy as np

from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.sea.latlon import haversine_km
from shiprouting.weather.forecast import Forecast


class Sea:
    """Sailing in one weather, at given speeds through the water."""

    def __init__(self, ship: Ship, weather: Forecast, departure, tables: str | None = None):
        self.ship, self.weather, self.departure, self.tables = ship, weather, np.datetime64(departure, "s"), tables
        self._costs = {}

    def cost(self, kn: float) -> FuelCost:
        if kn not in self._costs:
            speed = kn * KNOT
            waves = (AddedResistanceTable.between(self.tables, self.ship.seakeeping, speed) if self.tables
                     else AddedResistanceTable.build(self.ship.seakeeping, speed))
            self._costs[kn] = FuelCost(self.ship, self.weather, self.departure, speed, waves, frozen=False, currents=True)
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


def summary(lat, lon, kg, h, hours: float) -> dict:
    return {
        "km": round(float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()), 1),
        "hours": round(float(h.sum()), 1),
        "late_h": round(max(float(h.sum()) - hours, 0.0), 1),
        "fuel_t": round(float(kg.sum()) / 1000, 2),
        "lat": np.round(lat, 6).tolist(),
        "lon": np.round(lon, 6).tolist(),
    }


def track(lat, lon, on_link, planned_kn, kg, h, sailed_kn, hours: float) -> dict:
    """A sailed track as saved: its summary, and what recost needs to sail it again."""
    return summary(lat, lon, kg, h, hours) | {
        "on_link": np.asarray(on_link, bool).tolist(),
        "planned_kn": np.round(planned_kn, 6).tolist(),
        "segment_kn": np.round(sailed_kn, 6).tolist(),
        "segment_kg": np.round(kg, 3).tolist(),
        "segment_h": np.round(h, 6).tolist(),
    }


def recost(result: dict, sea: Sea, speeds_kn) -> dict:
    """Sail each track of a saved voyage ("shortest", "sailed") again in sea's weather, at its planned
    speeds. A track saved without planned speeds is sailed at the speeds it sailed.

    The track itself is kept: a voyage replanned on the way would have gone elsewhere in other
    weather, so this measures the same track, not the same strategy."""
    out = {}
    for name in ("shortest", "sailed"):
        t = result.get(name)
        if not t or "on_link" not in t:
            continue
        lat, lon = np.array(t["lat"]), np.array(t["lon"])
        kn = np.array(t.get("planned_kn", t["segment_kn"]), float)
        kg, h, sailed, _ = sea.sail(lat, lon, np.array(t["on_link"], bool), kn, speeds_kn, 0.0)
        out[name] = track(lat, lon, t["on_link"], kn, kg, h, sailed, result["hours"])
    return out


def on_time(result: dict, sea: Sea, speeds_kn, known: dict | None = None) -> dict:
    """Each track of a saved voyage ("shortest", "sailed") sailed again in sea's weather with its
    speeds chosen again (SpeedChoice.retime, of speeds_kn) for the least fuel that arrives by the
    voyage's deadline: the same track judged at the same arrival time, as a ship that keeps to its
    schedule would sail it. A track that cannot make the deadline is sailed at the fastest speed.
    The speeds are chosen knowing the weather ahead, so this is the fuel of the track, not of a
    strategy that has to guess. known: tracks already judged in this sea, by their points and
    deadline (filled in here), so a track two results share, such as the shortest route of the
    same departure, is judged once."""
    from shiprouting.route.speed import SpeedChoice
    from shiprouting.sea.latlon import SearchResult

    choice = SpeedChoice([sea.cost(k) for k in sorted(speeds_kn)])
    out = {}
    for name in ("shortest", "sailed"):
        t = result.get(name)
        if not t or "on_link" not in t:
            continue
        key = (tuple(t["lat"]), tuple(t["lon"]), tuple(t["on_link"]), result["hours"])
        if known is not None and key in known:
            out[name] = known[key]
            continue
        lat, lon, on_link = np.array(t["lat"]), np.array(t["lon"]), np.array(t["on_link"], bool)
        fastest = np.full(len(on_link), len(speeds_kn) - 1)
        r = SearchResult(lat, lon, on_link, [], 0.0, 0, 0.0, clock=np.zeros(len(lat)))
        clock, kg, _ = choice.sail(r, fastest)
        r = dataclasses.replace(r, clock=clock, cost=float(kg[fastest, np.arange(len(fastest))].sum()))
        try:  # the time a little later is worth about 2 x fuel / hours (fuel ~ distance^3 / hours^2)
            r = choice.retime(r, result["hours"], per_hour=2 * r.cost / result["hours"])
        except ValueError:
            pass  # not even the fastest speeds arrive in time: sailed at them
        k = choice.speed_index(r)
        _, kg_all, h_all = choice.sail(r, k)
        seg = np.arange(len(k))
        out[name] = track(lat, lon, on_link, np.array(sorted(speeds_kn), float)[k], kg_all[k, seg], h_all[k, seg],
                          np.array(sorted(speeds_kn), float)[k], result["hours"])
        if known is not None:
            known[key] = out[name]
    return out
