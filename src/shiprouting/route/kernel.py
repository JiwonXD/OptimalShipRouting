"""The fuel of moves, compiled with Numba: what FuelCost.kg computes, one move at a time.

It repeats, for scalars, the great-circle formulas (latlon.haversine_km,
quadtree.along), Forecast.sample, air_resistance and the lookups in the wave
and fuel-rate tables, for one or several speeds. Calling numpy on the handful of moves that leave one
cell costs far more than the arithmetic itself.
"""

import math

import numba
import numpy as np

from shiprouting.sea.latlon import EARTH_RADIUS_KM
from shiprouting.weather.conditions import T01_PER_TM10


@numba.njit(cache=True)
def _cell(grid, x):
    """Index of the grid interval holding x (clamped to the end intervals) and the weight of its upper end."""
    i = np.searchsorted(grid, x, side="right") - 1
    i = min(max(i, 0), len(grid) - 2)
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


@numba.njit(cache=True)
def sample(field, lat_grid, lon_grid, t_grid, lat, lon, t):
    """Forecast.sample for one point; t in seconds, inside t_grid."""
    k, wt = _cell(t_grid, t)
    dlat = lat_grid[1] - lat_grid[0]
    fy = (lat - lat_grid[0]) / dlat
    j = min(max(int(math.floor(fy)), 0), len(lat_grid) - 2)
    wy = min(max(fy - j, 0.0), 1.0)
    dlon = lon_grid[1] - lon_grid[0]
    fx = (lon - lon_grid[0]) / dlon
    i0 = int(math.floor(fx))
    wx = fx - i0
    i0 %= len(lon_grid)
    i1 = (i0 + 1) % len(lon_grid)  # wraps across the dateline
    num = den = 0.0
    for kk, wk in ((k, 1 - wt), (k + 1, wt)):
        for jj, wj in ((j, 1 - wy), (j + 1, wy)):
            for ii, wi in ((i0, 1 - wx), (i1, wx)):
                v = field[kk, jj, ii]
                if not math.isnan(v):
                    num += wk * wj * wi * v
                    den += wk * wj * wi
    return num / den if den > 0 else math.nan


@numba.njit(cache=True)
def _abs_angle(a):
    """|a| folded into 0..π."""
    return abs((a + math.pi) % (2 * math.pi) - math.pi)


@numba.njit(cache=True)
def fuel_kg(
    lat_a, lon_a, lat_b, lon_b, hours, out, counts,  # moves, the fuel out (speeds, moves), per speed (no waves, over MCR, after forecast)
    t0, frozen, speed, R_hull,  # departure s, weather frozen at t0?, per speed: m/s, calm-water resistance N
    wind_deg, C_DA, A_XV, rho_air,  # windage
    T01_grid, angle_grid, per_Hs2,  # AddedResistanceTable, per_Hs2 per speed
    R_grid, kg_h,  # per speed: fuel rate over total resistance, up to MCR
    lat_grid, lon_grid, t_grid, wind_u, wind_v, Hs_f, from_x, from_y, T_m10,  # forecast
):
    for m in range(len(lat_a)):
        p0, l0, p1, l1 = math.radians(lat_a[m]), math.radians(lon_a[m]), math.radians(lat_b[m]), math.radians(lon_b[m])
        h = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin((l1 - l0) / 2) ** 2
        km = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(max(h, 0.0), 1.0)))

        # Midpoint on the great circle, and the initial course.
        x = math.cos(p0) * math.cos(l0) + math.cos(p1) * math.cos(l1)
        y = math.cos(p0) * math.sin(l0) + math.cos(p1) * math.sin(l1)
        z = math.sin(p0) + math.sin(p1)
        lat, lon = math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))
        dl = l1 - l0
        heading = math.atan2(math.sin(dl) * math.cos(p1), math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl))

        # Every speed at the same midpoint: only the time it is reached differs.
        for s in range(len(speed)):
            t = t0
            if not frozen:
                t = t0 + (hours[m] + km / 2 / (speed[s] * 3.6)) * 3600
                if t > t_grid[-1]:
                    counts[s, 2] += 1
                    t = t_grid[-1]

            u = sample(wind_u, lat_grid, lon_grid, t_grid, lat, lon, t)
            v = sample(wind_v, lat_grid, lon_grid, t_grid, lat, lon, t)
            wind, wind_angle = math.hypot(u, v), math.atan2(-u, -v) - heading
            ax = -wind * math.cos(wind_angle) - speed[s]  # air velocity relative to the ship, ship axes
            ay = -wind * math.sin(wind_angle)
            psi = math.degrees(_abs_angle(math.atan2(-ay, -ax)))
            R = R_hull[s] + 0.5 * rho_air * np.interp(psi, wind_deg, C_DA) * A_XV * (ax * ax + ay * ay)

            Hs = sample(Hs_f, lat_grid, lon_grid, t_grid, lat, lon, t)
            if math.isnan(Hs):
                counts[s, 0] += 1
            elif Hs > 0:
                T01 = T01_PER_TM10 * sample(T_m10, lat_grid, lon_grid, t_grid, lat, lon, t)
                T01 = min(max(T01, T01_grid[0]), T01_grid[-1])
                wave_from = math.atan2(
                    sample(from_x, lat_grid, lon_grid, t_grid, lat, lon, t), sample(from_y, lat_grid, lon_grid, t_grid, lat, lon, t)
                )
                i, wt = _cell(T01_grid, T01)
                j, wa = _cell(angle_grid, _abs_angle(wave_from - heading))
                f = (1 - wt) * ((1 - wa) * per_Hs2[s, i, j] + wa * per_Hs2[s, i, j + 1]) + wt * (
                    (1 - wa) * per_Hs2[s, i + 1, j] + wa * per_Hs2[s, i + 1, j + 1]
                )
                R += Hs * Hs * f

            if R > R_grid[s, -1]:
                counts[s, 1] += 1
                out[s, m] = math.inf
            else:
                out[s, m] = np.interp(R, R_grid[s], kg_h[s]) * km / (speed[s] * 3.6)
