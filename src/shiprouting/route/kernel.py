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
def _interp(x, xp, fp):
    """np.interp for one x, without the array numba's np.interp builds for it."""
    if not x < xp[-1]:
        return fp[-1] if x >= xp[-1] else math.nan
    if x < xp[0]:
        return fp[0]
    j = np.searchsorted(xp, x, side="right") - 1
    if xp[j] == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    y = slope * (x - xp[j]) + fp[j]
    if math.isnan(y):
        y = slope * (x - xp[j + 1]) + fp[j + 1]
        if math.isnan(y) and fp[j] == fp[j + 1]:
            y = fp[j]
    return y


@numba.njit(cache=True)
def _move(
    s, km, heading, cu, cv, u, v, Hs, T_m10, from_x, from_y, counts,  # speed index, the move, the weather at its midpoint (cu, cv nan: no current)
    speed, R_hull, wind_deg, C_DA, A_XV, rho_air, T01_grid, angle_grid, per_Hs2, R_grid, kg_h,
):
    """The fuel, kg, and hours of one move at speed s in the weather given."""
    # Speed over the ground: the ship steers to stay on track, so the current across
    # the track takes part of the speed through the water and the current along it adds.
    sog = speed[s]
    if not (math.isnan(cu) or math.isnan(cv)):
        c_along = cu * math.sin(heading) + cv * math.cos(heading)
        c_across = cu * math.cos(heading) - cv * math.sin(heading)
        sog = math.sqrt(max(speed[s] ** 2 - c_across**2, 0.0)) + c_along
    if sog < 0.25 * speed[s]:  # held back by the current: not a move to take
        return math.inf, km / (speed[s] * 3.6)
    h = km / (sog * 3.6)

    wind, wind_angle = math.hypot(u, v), math.atan2(-u, -v) - heading
    ax = -wind * math.cos(wind_angle) - sog  # air velocity relative to the ship, ship axes
    ay = -wind * math.sin(wind_angle)
    psi = math.degrees(_abs_angle(math.atan2(-ay, -ax)))
    R = R_hull[s] + 0.5 * rho_air * _interp(psi, wind_deg, C_DA) * A_XV * (ax * ax + ay * ay)

    if math.isnan(Hs):
        counts[s, 0] += 1
    elif Hs > 0:
        T01 = T01_PER_TM10 * T_m10
        T01 = min(max(T01, T01_grid[0]), T01_grid[-1])
        wave_from = math.atan2(from_x, from_y)
        i, wt = _cell(T01_grid, T01)
        j, wa = _cell(angle_grid, _abs_angle(wave_from - heading))
        f = (1 - wt) * ((1 - wa) * per_Hs2[s, i, j] + wa * per_Hs2[s, i, j + 1]) + wt * (
            (1 - wa) * per_Hs2[s, i + 1, j] + wa * per_Hs2[s, i + 1, j + 1]
        )
        R += Hs * Hs * f

    if R > R_grid[s, -1]:
        counts[s, 1] += 1
        return math.inf, h
    return _interp(R, R_grid[s], kg_h[s]) * h, h


@numba.njit(cache=True)
def _geometry(lat_a, lon_a, lat_b, lon_b):
    """Length, km, midpoint on the great circle and initial course of a move."""
    p0, l0, p1, l1 = math.radians(lat_a), math.radians(lon_a), math.radians(lat_b), math.radians(lon_b)
    h = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin((l1 - l0) / 2) ** 2
    km = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(max(h, 0.0), 1.0)))
    x = math.cos(p0) * math.cos(l0) + math.cos(p1) * math.cos(l1)
    y = math.cos(p0) * math.sin(l0) + math.cos(p1) * math.sin(l1)
    z = math.sin(p0) + math.sin(p1)
    lat, lon = math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))
    dl = l1 - l0
    heading = math.atan2(math.sin(dl) * math.cos(p1), math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl))
    return km, lat, lon, heading


@numba.njit(cache=True)
def _when(s, t0, frozen, hours, km, speed, t_grid, counts):
    """The time, s, a move set out `hours` after t0 reaches its midpoint at speed s (held at the forecast's end)."""
    t = t0
    if not frozen:
        t = t0 + (hours + km / 2 / (speed[s] * 3.6)) * 3600
        if t > t_grid[-1]:
            counts[s, 2] += 1
            t = t_grid[-1]
    return t


@numba.njit(cache=True)
def fuel_kg(
    lat_a, lon_a, lat_b, lon_b, hours, out, out_h, counts,  # moves, the fuel and hours out (speeds, moves), per speed (no waves, over MCR, after forecast)
    t0, frozen, speed, R_hull,  # departure s, weather frozen at t0?, per speed: m/s, calm-water resistance N
    wind_deg, C_DA, A_XV, rho_air,  # windage
    T01_grid, angle_grid, per_Hs2,  # AddedResistanceTable, per_Hs2 per speed
    R_grid, kg_h,  # per speed: fuel rate over total resistance, up to MCR
    lat_grid, lon_grid, t_grid, wind_u, wind_v, Hs_f, from_x, from_y, T_m10,  # forecast
    current_u, current_v, currents,  # surface current, and whether to use it
):
    for m in range(len(lat_a)):
        km, lat, lon, heading = _geometry(lat_a[m], lon_a[m], lat_b[m], lon_b[m])
        # Every speed at the same midpoint: only the time it is reached differs.
        for s in range(len(speed)):
            t = _when(s, t0, frozen, hours[m], km, speed, t_grid, counts)
            cu = cv = math.nan
            if currents:
                cu = sample(current_u, lat_grid, lon_grid, t_grid, lat, lon, t)
                cv = sample(current_v, lat_grid, lon_grid, t_grid, lat, lon, t)
            out[s, m], out_h[s, m] = _move(
                s, km, heading, cu, cv,
                sample(wind_u, lat_grid, lon_grid, t_grid, lat, lon, t), sample(wind_v, lat_grid, lon_grid, t_grid, lat, lon, t),
                sample(Hs_f, lat_grid, lon_grid, t_grid, lat, lon, t), sample(T_m10, lat_grid, lon_grid, t_grid, lat, lon, t),
                sample(from_x, lat_grid, lon_grid, t_grid, lat, lon, t), sample(from_y, lat_grid, lon_grid, t_grid, lat, lon, t),
                counts, speed, R_hull, wind_deg, C_DA, A_XV, rho_air, T01_grid, angle_grid, per_Hs2, R_grid, kg_h,
            )  # fmt: skip


@numba.njit(cache=True)
def _steps(field, j, wy, i0, i1, wx, num, den):
    """For one point, the weighted sum of field's non-nan corners and their weight, at every forecast step."""
    for k in range(field.shape[0]):
        n = d = 0.0
        for jj, wj in ((j, 1 - wy), (j + 1, wy)):
            for ii, wi in ((i0, 1 - wx), (i1, wx)):
                v = field[k, jj, ii]
                if not math.isnan(v):
                    n += wj * wi * v
                    d += wj * wi
        num[k], den[k] = n, d


@numba.njit(cache=True)
def _at(num, den, k, wt):
    """The field between forecast steps k and k + 1 (weight wt on k + 1), from _steps' sums:
    sample() but for the order of the additions."""
    d = (1 - wt) * den[k] + wt * den[k + 1]
    return ((1 - wt) * num[k] + wt * num[k + 1]) / d if d > 0 else math.nan


@numba.njit(cache=True, parallel=True)
def fuel_kg_times(
    lat_a, lon_a, lat_b, lon_b, hours, out, out_h, counts,  # one move set out at each of hours; out (speeds, len(hours))
    t0, frozen, speed, R_hull, wind_deg, C_DA, A_XV, rho_air, T01_grid, angle_grid, per_Hs2, R_grid, kg_h,
    lat_grid, lon_grid, t_grid, wind_u, wind_v, Hs_f, from_x, from_y, T_m10, current_u, current_v, currents,
):
    """fuel_kg for one move at many times, as SpeedChoice.retime asks: the place and its weights in
    the forecast grid are worked out once, and each field's corners summed once per forecast step."""
    km, lat, lon, heading = _geometry(lat_a, lon_a, lat_b, lon_b)
    fy = (lat - lat_grid[0]) / (lat_grid[1] - lat_grid[0])
    j = min(max(int(math.floor(fy)), 0), len(lat_grid) - 2)
    wy = min(max(fy - j, 0.0), 1.0)
    fx = (lon - lon_grid[0]) / (lon_grid[1] - lon_grid[0])
    i0 = int(math.floor(fx))
    wx = fx - i0
    i0 %= len(lon_grid)
    i1 = (i0 + 1) % len(lon_grid)  # wraps across the dateline
    nt = len(t_grid)
    num, den = np.empty((8, nt)), np.empty((8, nt))
    _steps(wind_u, j, wy, i0, i1, wx, num[0], den[0])
    _steps(wind_v, j, wy, i0, i1, wx, num[1], den[1])
    _steps(Hs_f, j, wy, i0, i1, wx, num[2], den[2])
    _steps(T_m10, j, wy, i0, i1, wx, num[3], den[3])
    _steps(from_x, j, wy, i0, i1, wx, num[4], den[4])
    _steps(from_y, j, wy, i0, i1, wx, num[5], den[5])
    if currents:
        _steps(current_u, j, wy, i0, i1, wx, num[6], den[6])
        _steps(current_v, j, wy, i0, i1, wx, num[7], den[7])
    n, S = len(hours), len(speed)
    tally = np.zeros((n, S, 3), np.int64)  # counts per time, added up after the threads are done
    for m in numba.prange(n):
        for s in range(S):
            k, wt = _cell(t_grid, _when(s, t0, frozen, hours[m], km, speed, t_grid, tally[m]))
            cu = cv = math.nan
            if currents:
                cu, cv = _at(num[6], den[6], k, wt), _at(num[7], den[7], k, wt)
            out[s, m], out_h[s, m] = _move(
                s, km, heading, cu, cv, _at(num[0], den[0], k, wt), _at(num[1], den[1], k, wt), _at(num[2], den[2], k, wt),
                _at(num[3], den[3], k, wt), _at(num[4], den[4], k, wt), _at(num[5], den[5], k, wt), tally[m], speed, R_hull,
                wind_deg, C_DA, A_XV, rho_air, T01_grid, angle_grid, per_Hs2, R_grid, kg_h,
            )  # fmt: skip
    for m in range(n):
        for s in range(S):
            for c in range(3):
                counts[s, c] += tally[m, s, c]


@numba.njit(cache=True)
def prune(t, f, kg, h, left_h, end, dt_hours, price):
    """One segment of SpeedChoice.retime: every label (t, f) going on at every speed (kg, h of
    shape (speeds, labels)); those that can still arrive by end, with left_h hours to go at the
    least, kept if they are the least fuel, or the least fuel + price × hours, among the labels
    in their bucket of dt_hours (ties to the earlier, then to the first). Returns the times,
    fuels, parent labels and speeds of those kept, in the order speed by speed, label by label."""
    S, L = kg.shape
    n = S * L
    t2, f2 = np.empty(n), np.empty(n)
    ok = np.zeros(n, np.bool_)
    lo, hi = np.inf, -np.inf
    for s in range(S):
        for i in range(L):
            c = s * L + i
            t2[c], f2[c] = t[i] + h[s, i], f[i] + kg[s, i]
            if math.isfinite(f2[c]) and t2[c] + left_h <= end:
                ok[c] = True
                b = math.floor(t2[c] / dt_hours)
                lo, hi = min(lo, b), max(hi, b)
    keep = np.zeros(n, np.bool_)
    if lo > hi:
        return t2[:0], f2[:0], np.zeros(0, np.int64), np.zeros(0, np.int64)
    nb = int(hi - lo) + 1
    for by_price in (False, True):
        best = np.full(nb, -1, np.int64)
        for c in range(n):
            if not ok[c]:
                continue
            b = int(math.floor(t2[c] / dt_hours) - lo)
            v = f2[c] + price * t2[c] if by_price else f2[c]
            e = best[b]
            if e < 0:
                best[b] = c
            else:
                ve = f2[e] + price * t2[e] if by_price else f2[e]
                if v < ve or (v == ve and t2[c] < t2[e]):
                    best[b] = c
        for b in range(nb):
            if best[b] >= 0:
                keep[best[b]] = True
    idx = np.nonzero(keep)[0]
    return t2[idx], f2[idx], idx % L, idx // L


@numba.njit(cache=True)
def _less(hc, hn, a, b):
    """Heap entry a before b: by cost, then by node, as heapq orders (cost, node) tuples."""
    return hc[a] < hc[b] or (hc[a] == hc[b] and hn[a] < hn[b])


@numba.njit(cache=True)
def _push(hc, hn, size, cost, node):
    i = size
    hc[i], hn[i] = cost, node
    while i > 0:
        p = (i - 1) // 2
        if not _less(hc, hn, i, p):
            break
        hc[i], hc[p], hn[i], hn[p] = hc[p], hc[i], hn[p], hn[i]
        i = p
    return size + 1


@numba.njit(cache=True)
def _pop(hc, hn, size):
    cost, node = hc[0], hn[0]
    size -= 1
    hc[0], hn[0] = hc[size], hn[size]
    i = 0
    while True:
        a, b, m = 2 * i + 1, 2 * i + 2, i
        if a < size and _less(hc, hn, a, m):
            m = a
        if b < size and _less(hc, hn, b, m):
            m = b
        if m == i:
            break
        hc[i], hc[m], hn[i], hn[m] = hc[m], hc[i], hn[m], hn[i]
        i = m
    return cost, node, size


@numba.njit(cache=True)
def search(
    indptr, indices, glat, glon, jptr, jto, jkg, jh,  # the graph, and its links as jumps per node with kg and hours per speed
    g, price, budget, best, clock, parent, via, closed, hc, hn, state, counts,  # goal, price of an hour, the search's state
    t0, frozen, speed, R_hull, wind_deg, C_DA, A_XV, rho_air, T01_grid, angle_grid, per_Hs2, R_grid, kg_h,
    lat_grid, lon_grid, t_grid, wind_u, wind_v, Hs_f, from_x, from_y, T_m10, current_u, current_v, currents,
):
    """quadtree.astar with SpeedChoice.edge_cost and link_cost and no heuristic, compiled: each
    move at the speed with the least fuel + price × hours, at the time its cell is reached.
    Expands up to `budget` cells and returns 0, or 1 on reaching g, 2 when the heap needs more
    room, 3 when nothing is left to expand; state holds the heap's size and the cells expanded."""
    size, expanded = state[0], state[1]
    room = 0
    for c in range(len(indptr) - 1):
        room = max(room, indptr[c + 1] - indptr[c] + jptr[c + 1] - jptr[c])
    done, status = 0, 3
    while size > 0:
        if done >= budget:
            status = 0
            break
        if size + room > len(hc):
            status = 2
            break
        _, c, size = _pop(hc, hn, size)
        if closed[c]:
            continue
        closed[c] = True
        expanded += 1
        done += 1
        if c == g:
            status = 1
            break
        for e in range(indptr[c], indptr[c + 1]):
            n = indices[e]
            if closed[n]:
                continue
            km, lat, lon, heading = _geometry(glat[c], glon[c], glat[n], glon[n])
            w, dt, usable = math.inf, math.nan, True
            for s in range(len(speed)):
                t = _when(s, t0, frozen, clock[c], km, speed, t_grid, counts)
                cu = cv = math.nan
                if currents:
                    cu = sample(current_u, lat_grid, lon_grid, t_grid, lat, lon, t)
                    cv = sample(current_v, lat_grid, lon_grid, t_grid, lat, lon, t)
                kg, h = _move(
                    s, km, heading, cu, cv,
                    sample(wind_u, lat_grid, lon_grid, t_grid, lat, lon, t), sample(wind_v, lat_grid, lon_grid, t_grid, lat, lon, t),
                    sample(Hs_f, lat_grid, lon_grid, t_grid, lat, lon, t), sample(T_m10, lat_grid, lon_grid, t_grid, lat, lon, t),
                    sample(from_x, lat_grid, lon_grid, t_grid, lat, lon, t), sample(from_y, lat_grid, lon_grid, t_grid, lat, lon, t),
                    counts, speed, R_hull, wind_deg, C_DA, A_XV, rho_air, T01_grid, angle_grid, per_Hs2, R_grid, kg_h,
                )  # fmt: skip
                total = kg + price * h
                if math.isnan(total):  # np.argmin would take it, and no cost is less than nan
                    usable = False
                    break
                if s == 0 or total < w:
                    w, dt = total, h
            cost = best[c] + w
            if usable and cost < best[n]:
                best[n], clock[n], parent[n], via[n] = cost, clock[c] + dt, c, -1
                size = _push(hc, hn, size, cost, n)
        for j in range(jptr[c], jptr[c + 1]):
            n = jto[j]
            if closed[n]:
                continue
            w, dt = jkg[j, 0] + price * jh[j, 0], jh[j, 0]
            for s in range(1, len(speed)):
                total = jkg[j, s] + price * jh[j, s]
                if total < w:
                    w, dt = total, jh[j, s]
            cost = best[c] + w
            if cost < best[n]:
                best[n], clock[n], parent[n], via[n] = cost, clock[c] + dt, c, j
                size = _push(hc, hn, size, cost, n)
    state[0], state[1] = size, expanded
    return status
