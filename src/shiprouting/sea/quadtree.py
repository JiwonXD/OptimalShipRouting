"""Sea routing on a grid that is fine only along the coast.

The open ocean keeps the 0.25° cells of the latitude/longitude grid. A cell
that is partly land is split into four, again and again down to `min_step`,
so narrow straits, canal entrances and ports get small cells without making
the whole ocean fine. Every cell is a node; cells whose boxes touch (along a
side or at a corner) are joined by an edge when the straight line between
their centres misses land. A* runs on that graph as on the regular grid.
"""

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from shiprouting.sea.land import Land
from shiprouting.sea.latlon import CHECKPOINTS, ICE_FREE, ROUTES, SearchResult, haversine_km, segments_on_land


@dataclass(frozen=True)
class Graph:
    lat: np.ndarray  # cell centres
    lon: np.ndarray
    size: np.ndarray  # cell side, degrees
    indptr: np.ndarray  # edges of node n: indices[indptr[n]:indptr[n + 1]]
    indices: np.ndarray
    km: np.ndarray  # length of each edge
    links: tuple = ()  # (node at the first end, node at the last end, Link)

    @classmethod
    def build(cls, land: Land, root: float = 0.25, min_step: float = 1 / 32, lat_limits=ICE_FREE, links=()) -> "Graph":
        x, y, s = split_coast(land, root, min_step, lat_limits)
        lon, lat = x + s / 2, y + s / 2
        water = ~land.contains(lat, lon)
        x, y, s, lat, lon = x[water], y[water], s[water], lat[water], lon[water]

        i, j, km = edges(land, x, y, s, lat, lon)
        # Keep only bodies of water that hold at least one cell of root size: the split cells
        # of a pocket behind a breakwater, or a cell with no open edge, lead nowhere.
        n, label = connected_components(coo_matrix((km, (i, j)), shape=(len(lat), len(lat))), directed=False)
        sea = np.zeros(n, bool)
        sea[label[s >= root]] = True
        keep = sea[label] & (np.bincount(np.concatenate([i, j]), minlength=len(lat)) > 0)
        new = np.cumsum(keep) - 1
        e = keep[i] & keep[j]
        i, j, km = new[i[e]], new[j[e]], km[e]
        lat, lon, s = lat[keep], lon[keep], s[keep]

        a, b = np.concatenate([i, j]), np.concatenate([j, i])
        order = np.argsort(a, kind="stable")
        indptr = np.searchsorted(a[order], np.arange(len(lat) + 1))
        graph = cls(lat, lon, s, indptr, b[order], np.concatenate([km, km])[order])

        joined = []
        for k in links:
            ends = join(land, graph, k.lat[0], k.lon[0]), join(land, graph, k.lat[-1], k.lon[-1])
            if None not in ends:
                joined.append((*ends, k))
        return cls(lat, lon, s, graph.indptr, graph.indices, graph.km, tuple(joined))

    def nearest(self, lat: float, lon: float) -> int:
        return int(np.argmin(haversine_km(self.lat, self.lon, lat, lon)))


def split_coast(land: Land, root: float, min_step: float, lat_limits):
    """Corners (lon, lat) and sides of the cells: 0.25° boxes, with every box that is partly land split down to min_step."""
    y, x = np.meshgrid(np.arange(lat_limits[0], lat_limits[1], root), np.arange(-180.0, 180.0, root), indexing="ij")
    x, y, s = x.ravel(), y.ravel(), np.full(x.size, root)
    leaves = []
    while len(x):
        mixed = np.zeros(len(x), bool)
        if s[0] / 2 >= min_step * (1 - 1e-9):
            boxes = shapely.box(x, y, x + s, y + s)
            touch = np.unique(land.tree.query(boxes, predicate="intersects")[0])
            f = land.fraction(boxes[touch])
            mixed[touch[(f > 0) & (f < 1 - 1e-9)]] = True
        leaves.append((x[~mixed], y[~mixed], s[~mixed]))
        h = s[mixed] / 2
        x = np.concatenate([x[mixed], x[mixed] + h, x[mixed], x[mixed] + h])
        y = np.concatenate([y[mixed], y[mixed], y[mixed] + h, y[mixed] + h])
        s = np.concatenate([h, h, h, h])
    return (np.concatenate(v) for v in zip(*leaves))


def edges(land: Land, x, y, s, lat, lon):
    """Pairs of touching cells whose centre-to-centre line misses land, and their lengths."""
    boxes = shapely.box(x, y, x + s, y + s)
    tree = shapely.STRtree(boxes)
    i, j = tree.query(boxes, predicate="intersects")
    keep = i < j
    i, j = i[keep], j[keep]
    shift = np.zeros(len(i))
    west = np.nonzero(x <= -180.0)[0]  # cells on the -180° edge also touch those on the +180° edge
    wi, wj = tree.query(shapely.box(x[west] + 360, y[west], x[west] + s[west] + 360, y[west] + s[west]), predicate="intersects")
    i, j = np.concatenate([i, west[wi]]), np.concatenate([j, wj])
    shift = np.concatenate([shift, np.full(len(wi), 360.0)])

    a = np.column_stack([lon[i] + shift, lat[i]])
    b = np.column_stack([lon[j], lat[j]])
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    hit = land.crosses(lines)
    wrapped = shift != 0
    if wrapped.any():
        hit[wrapped] |= land.crosses(shapely.transform(lines[wrapped], lambda xy: xy - [360.0, 0.0]))
    i, j = i[~hit], j[~hit]
    return i, j, haversine_km(lat[i], lon[i], lat[j], lon[j])


def join(land: Land, graph: Graph, lat: float, lon: float, tries: int = 100):
    """The nearest node whose straight hop to (lat, lon) misses land, or None."""
    near = np.argsort(haversine_km(graph.lat, graph.lon, lat, lon))[:tries]
    a = np.column_stack([graph.lon[near], graph.lat[near]])
    b = np.broadcast_to([lon, lat], a.shape)
    clear = ~land.crosses(shapely.linestrings(np.stack([a, b], axis=1)))
    return int(near[np.argmax(clear)]) if clear.any() else None


def astar(graph: Graph, s: int, g: int) -> SearchResult:
    t0 = time.perf_counter()

    def h(n) -> float:
        return float(haversine_km(graph.lat[n], graph.lon[n], graph.lat[g], graph.lon[g]))

    jumps = {}  # node -> [(node, km, link, reversed)]
    for a, b, link in graph.links:
        km = (
            float(haversine_km(graph.lat[a], graph.lon[a], link.lat[0], link.lon[0]))
            + link.km
            + float(haversine_km(graph.lat[b], graph.lon[b], link.lat[-1], link.lon[-1]))
        )
        jumps.setdefault(a, []).append((b, km, link, False))
        jumps.setdefault(b, []).append((a, km, link, True))

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
        lo, hi = graph.indptr[c], graph.indptr[c + 1]
        steps = [(int(n), float(km), None, False) for n, km in zip(graph.indices[lo:hi], graph.km[lo:hi])]
        for n, km, link, rev in steps + jumps.get(c, []):
            cost = best[c] + km
            if n not in closed and cost < best.get(n, math.inf):
                best[n] = cost
                parent[n] = (c, link, rev)
                heapq.heappush(open_list, (cost + h(n), n))
    else:
        raise ValueError(f"no route ({expanded} cells expanded)")

    nodes, via = [g], []  # from goal back to start, each with the link that led into it
    while parent[nodes[-1]] is not None:
        c, link, rev = parent[nodes[-1]]
        via.append((link, rev))
        nodes.append(c)
    lat, lon = [graph.lat[s]], [graph.lon[s]]
    on_link, used = [], []
    for n, (link, rev) in zip(reversed(nodes[:-1]), reversed(via)):
        if link is not None:
            pts_lat, pts_lon = (link.lat[::-1], link.lon[::-1]) if rev else (link.lat, link.lon)
            lat += list(pts_lat)
            lon += list(pts_lon)
            on_link += [False] + [True] * (len(pts_lat) - 1)
            used.append(link.name)
        lat.append(graph.lat[n])
        lon.append(graph.lon[n])
        on_link.append(False)
    return SearchResult(np.array(lat), np.array(lon), np.array(on_link, bool), used, best[g], expanded, time.perf_counter() - t0)


def run(land: Land, min_step: float, links=()) -> list[dict]:
    t0 = time.perf_counter()
    graph = Graph.build(land, min_step=min_step, links=links)
    build_s = time.perf_counter() - t0
    sizes = ", ".join(f"{s:g}°: {n}" for s, n in zip(*np.unique(graph.size, return_counts=True)))
    print(f"min step {min_step:g}°: {len(graph.lat)} cells ({sizes}), {len(graph.indices) // 2} edges, built in {build_s:.0f} s")
    for name in {k.name for k in links} - {k.name for _, _, k in graph.links}:
        print(f"link {name}: no cell nearby with a clear hop to an end; left out")
    for a, b, link in graph.links:
        hops = [float(haversine_km(graph.lat[n], graph.lon[n], link.lat[e], link.lon[e])) for n, e in ((a, 0), (b, -1))]
        print(f"link {link.name}: {link.km:.1f} km, ends joined with hops of {hops[0]:.1f} km and {hops[1]:.1f} km")
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {
            "route": name,
            "min_step": min_step,
            "snap_km": round(float(max(haversine_km(*start, graph.lat[s], graph.lon[s]), haversine_km(*goal, graph.lat[g], graph.lon[g]))), 1),
            "great_circle_km": round(float(haversine_km(*start, *goal))),
        }
        try:
            r = astar(graph, s, g)
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
                lat=np.round(r.lat, 4).tolist(),
                lon=np.round(r.lon, 4).tolist(),
            )
        print({k: v for k, v in row.items() if k not in ("lat", "lon")}, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import json

    from shiprouting.sea.land import load_osm_land_polygons

    parser = argparse.ArgumentParser(description="Run the test voyages on the coast-refined grid with A*.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, action="append", help="smallest cell side in degrees (repeatable)")
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    args = parser.parse_args()
    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    rows = [row for step in args.min_step or [1 / 32] for row in run(land, step, links)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
