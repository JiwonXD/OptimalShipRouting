"""First attempt at sea routing: a regular latitude/longitude grid and A*.

A cell is water when its centre is not on land. The ship moves to any of the
8 neighbouring water cells as long as the straight line between the two
centres does not touch land; the cost of a move is the great-circle distance
between the cell centres, and A* uses the great-circle distance to the goal
as its heuristic. Canals too narrow for any cell are added as links that join
at each end the nearest water cell with a clear straight hop to it.
"""

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from shiprouting.sea.land import Land

EARTH_RADIUS_KM = 6371.0088
MOVES = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
# Sea ice is not modelled yet; as a stand-in, cells outside these latitudes
# count as blocked. 72°N keeps the Barents Sea and the Bering Strait open but
# closes the Arctic passages; 60°S keeps the Drake Passage open.
ICE_FREE = (-60.0, 72.0)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class LatLonGrid:
    step: float  # cell size, degrees
    water: np.ndarray  # (rows, cols) bool; row 0 is the southernmost
    moves: np.ndarray  # (rows, cols, 8) bool; MOVES[k] from the cell is open
    links: tuple = ()  # (cell at the first end, cell at the last end, Link)

    @classmethod
    def build(cls, land: Land, step: float, lat_limits: tuple[float, float] = ICE_FREE, links=()) -> "LatLonGrid":
        lat = -90 + step / 2 + step * np.arange(round(180 / step))
        lon = -180 + step / 2 + step * np.arange(round(360 / step))
        lon2, lat2 = np.meshgrid(lon, lat)
        water = ~land.contains(lat2.ravel(), lon2.ravel()).reshape(lat2.shape)
        water &= (lat2 >= lat_limits[0]) & (lat2 <= lat_limits[1])
        moves = open_moves(land, water, lat2, lon2, step)
        # A water cell with no open move cannot be reached or left; a port snapped onto one has no route.
        water &= moves.any(axis=2)
        grid = cls(step, water, moves)
        joined = []
        for k in links:
            a, b = join(land, grid, k.lat[0], k.lon[0]), join(land, grid, k.lat[-1], k.lon[-1])
            if a is not None and b is not None:
                joined.append((grid.cell(*a), grid.cell(*b), k))
        return cls(step, water, moves, tuple(joined))

    def center(self, i: int, j: int) -> tuple[float, float]:
        return -90 + self.step * (i + 0.5), -180 + self.step * (j + 0.5)

    def cell(self, lat: float, lon: float) -> tuple[int, int]:
        return int((lat + 90) // self.step), int((lon + 180) // self.step)


def open_moves(land: Land, water, lat2, lon2, step: float) -> np.ndarray:
    """For every cell and move, whether both ends are water and the line between them misses land."""
    import shapely

    rows, cols = water.shape
    moves = np.zeros((rows, cols, len(MOVES)), bool)
    for k, (di, dj) in enumerate(MOVES):
        if (-di, -dj) in MOVES[:k]:
            continue  # same lines as the opposite move, filled in below
        ends = np.zeros_like(water)
        src = (slice(max(0, -di), rows - max(0, di)), slice(None))
        dst = (slice(max(0, di), rows - max(0, -di)), slice(None))
        ends[src] = water[src] & np.roll(water, -dj, axis=1)[dst]
        i, j = np.nonzero(ends)
        a = np.column_stack([lon2[i, j], lat2[i, j]])
        b = a + [dj * step, di * step]  # may run past ±180°; land beyond is checked shifted back
        lines = shapely.linestrings(np.stack([a, b], axis=1))
        hit = land.crosses(lines)
        for shift, past in ((-360.0, b[:, 0] > 180), (360.0, b[:, 0] < -180)):
            if past.any():
                hit[past] |= land.crosses(shapely.transform(lines[past], lambda xy: xy + [shift, 0]))
        ok = ~hit
        moves[i, j, k] = ok
        moves[i + di, (j + dj) % cols, MOVES.index((-di, -dj))] = ok
    return moves


@dataclass(frozen=True)
class SearchResult:
    lat: np.ndarray  # cell centres from start to goal, with the points of any link in between
    lon: np.ndarray
    on_link: np.ndarray  # bool per segment: part of a link's own polyline
    links: list  # names of the links used
    km: float
    expanded: int  # cells taken off the open list
    seconds: float


def astar(grid: LatLonGrid, start: tuple[float, float], goal: tuple[float, float]) -> SearchResult:
    t0 = time.perf_counter()
    cols = grid.water.shape[1]
    s, g = grid.cell(*start), grid.cell(*goal)
    for name, c in (("start", s), ("goal", g)):
        if not grid.water[c]:
            raise ValueError(f"{name} is on land")
    goal_lat, goal_lon = grid.center(*g)

    def dist(a, b) -> float:
        return float(haversine_km(*grid.center(*a), *grid.center(*b)))

    def h(c) -> float:
        return float(haversine_km(*grid.center(*c), goal_lat, goal_lon))

    jumps = {}  # cell -> [(cell, km, link, reversed)]
    for a, b, link in grid.links:
        km_a = float(haversine_km(*grid.center(*a), link.lat[0], link.lon[0]))
        km_b = float(haversine_km(*grid.center(*b), link.lat[-1], link.lon[-1]))
        jumps.setdefault(a, []).append((b, km_a + link.km + km_b, link, False))
        jumps.setdefault(b, []).append((a, km_a + link.km + km_b, link, True))

    best = {s: 0.0}
    parent = {s: None}
    closed = set()
    open_list = [(h(s), s)]
    expanded = 0
    while open_list:
        _, c = heapq.heappop(open_list)
        if c in closed:
            continue
        closed.add(c)
        expanded += 1
        if c == g:
            break
        for k, (di, dj) in enumerate(MOVES):
            n = (c[0] + di, (c[1] + dj) % cols)  # longitude wraps at ±180°
            if not grid.moves[c][k] or n in closed:
                continue
            cost = best[c] + dist(c, n)
            if cost < best.get(n, math.inf):
                best[n] = cost
                parent[n] = (c, None, False)
                heapq.heappush(open_list, (cost + h(n), n))
        for n, km, link, rev in jumps.get(c, ()):
            cost = best[c] + km
            if n not in closed and cost < best.get(n, math.inf):
                best[n] = cost
                parent[n] = (c, link, rev)
                heapq.heappush(open_list, (cost + h(n), n))
    else:
        raise ValueError(f"no route ({expanded} cells expanded)")

    steps = [g]  # cells from goal back to start, each with the link that led into it
    via = []
    while parent[steps[-1]] is not None:
        c, link, rev = parent[steps[-1]]
        via.append((link, rev))
        steps.append(c)
    lat, lon = [grid.center(*s)[0]], [grid.center(*s)[1]]
    on_link, used = [], []
    for c, (link, rev) in zip(reversed(steps[:-1]), reversed(via)):
        if link is not None:
            pts_lat, pts_lon = (link.lat[::-1], link.lon[::-1]) if rev else (link.lat, link.lon)
            lat += list(pts_lat)
            lon += list(pts_lon)
            on_link += [False] + [True] * (len(pts_lat) - 1)
            used.append(link.name)
        lat.append(grid.center(*c)[0])
        lon.append(grid.center(*c)[1])
        on_link.append(False)
    return SearchResult(np.array(lat), np.array(lon), np.array(on_link, bool), used, best[g], expanded, time.perf_counter() - t0)


# Test voyages (start, goal) and the straits or canals a sensible route would pass.
ROUTES = {
    "부산→로테르담": ((35.05, 129.10), (51.98, 3.95)),
    "부산→LA": ((35.05, 129.10), (33.70, -118.27)),
    "흑해→지중해": ((43.0, 34.0), (36.0, 15.0)),
    "뉴욕→LA": ((40.45, -73.80), (33.70, -118.27)),
    "지브롤터 통과": ((36.5, -8.0), (36.0, 0.0)),
    "도버 통과": ((50.5, -1.0), (51.5, 2.5)),
    "싱가포르 통과": ((1.0, 103.0), (1.5, 105.0)),
}
CHECKPOINTS = {
    "싱가포르": (1.2, 103.8),
    "바브엘만데브": (12.6, 43.3),
    "수에즈": (30.5, 32.35),
    "지브롤터": (35.95, -5.6),
    "도버": (51.0, 1.5),
    "보스포루스": (41.1, 29.05),
    "다다넬스": (40.2, 26.4),
    "파나마": (9.1, -79.7),
    "희망봉": (-34.8, 20.0),
    "케이프혼": (-56.5, -67.0),
}


def nearest_water(grid: LatLonGrid, lat: float, lon: float) -> tuple[float, float]:
    """Centre of the nearest water cell (a port often lies on a land cell)."""
    i, j = np.nonzero(grid.water)
    c_lat, c_lon = -90 + grid.step * (i + 0.5), -180 + grid.step * (j + 0.5)
    k = int(np.argmin(haversine_km(c_lat, c_lon, lat, lon)))
    return float(c_lat[k]), float(c_lon[k])


def join(land: Land, grid: LatLonGrid, lat: float, lon: float, tries: int = 100):
    """Centre of the nearest water cell whose straight hop to (lat, lon) misses land, or None."""
    import shapely

    i, j = np.nonzero(grid.water)
    c_lat, c_lon = -90 + grid.step * (i + 0.5), -180 + grid.step * (j + 0.5)
    near = np.argsort(haversine_km(c_lat, c_lon, lat, lon))[:tries]
    a = np.column_stack([c_lon[near], c_lat[near]])
    b = np.broadcast_to([lon, lat], a.shape)
    clear = ~land.crosses(shapely.linestrings(np.stack([a, b], axis=1)))
    if not clear.any():
        return None
    k = near[np.argmax(clear)]
    return float(c_lat[k]), float(c_lon[k])


def segments_on_land(land: Land, lat, lon, skip=None) -> int:
    """Moves whose straight line between points touches land, leaving out the segments marked in `skip`."""
    import shapely

    lon = np.degrees(np.unwrap(np.radians(lon)))  # continuous across ±180°
    hit = np.zeros(len(lat) - 1, bool)
    for shift in (0.0, -360.0, 360.0):
        a = np.column_stack([lon[:-1] + shift, lat[:-1]])
        b = np.column_stack([lon[1:] + shift, lat[1:]])
        hit |= land.crosses(shapely.linestrings(np.stack([a, b], axis=1)))
    if skip is not None:
        hit &= ~skip
    return int(hit.sum())


def run(land: Land, step: float, links=()) -> list[dict]:
    t0 = time.perf_counter()
    grid = LatLonGrid.build(land, step, links=links)
    build_s = time.perf_counter() - t0
    print(f"grid {step}°: {grid.water.shape[0]}x{grid.water.shape[1]}, {int(grid.water.sum())} water cells, built in {build_s:.0f} s")
    for name in {k.name for k in links} - {k.name for _, _, k in grid.links}:
        print(f"link {name}: no water cell nearby with a clear hop to an end; left out")
    for a, b, link in grid.links:
        ends = (grid.center(*a), grid.center(*b))
        connect = [float(haversine_km(*e, la, lo)) for e, la, lo in zip(ends, link.lat[[0, -1]], link.lon[[0, -1]])]
        print(f"link {link.name}: {link.km:.1f} km, ends joined to cells {ends} ({connect[0]:.1f} km, {connect[1]:.1f} km)")
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = nearest_water(grid, *start), nearest_water(grid, *goal)
        row = {
            "route": name,
            "step": step,
            "snap_km": round(float(max(haversine_km(*start, *s), haversine_km(*goal, *g))), 1),
            "great_circle_km": round(float(haversine_km(*start, *goal))),
        }
        try:
            r = astar(grid, s, g)
        except ValueError as e:
            row["error"] = str(e)
        else:
            near = [n for n, (la, lo) in CHECKPOINTS.items() if haversine_km(r.lat, r.lon, la, lo).min() < 50]
            row.update(
                km=round(r.km),
                expanded=r.expanded,
                seconds=round(r.seconds, 1),
                cells=len(r.lat),
                via=near,
                links=r.links,
                moves_touching_land=segments_on_land(land, r.lat, r.lon, skip=r.on_link),
                lat=np.round(r.lat, 3).tolist(),
                lon=np.round(r.lon, 3).tolist(),
            )
        print({k: v for k, v in row.items() if k not in ("lat", "lon")}, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import json

    from shiprouting.sea.land import load_osm_land_polygons

    parser = argparse.ArgumentParser(description="Run the test voyages on the lat/lon grid with A*.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--step", type=float, action="append", help="cell size in degrees (repeatable)")
    parser.add_argument("--canals", help="folder with the Overpass canal files (see links.CANALS)")
    args = parser.parse_args()
    land = load_osm_land_polygons(args.land)
    links = []
    if args.canals:
        from shiprouting.sea.links import load_canals

        links = load_canals(args.canals)
    rows = [row for step in args.step or [0.25] for row in run(land, step, links)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
