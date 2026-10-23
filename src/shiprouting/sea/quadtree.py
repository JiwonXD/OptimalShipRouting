"""Sea routing on a grid that is fine only along the coast.

The open ocean keeps the 0.25° cells of the latitude/longitude grid. A cell
that is partly land is split into four, again and again down to `min_step`,
so narrow straits, canal entrances and ports get small cells without making
the whole ocean fine. Every cell is a node; cells whose boxes touch (along a
side or at a corner) are joined by an edge when the straight line between
their centres misses land. A* runs on that graph as on the regular grid, and
the path it finds is then straightened by cutting corners where the straight
line misses land.
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
from shiprouting.sea.latlon import CHECKPOINTS, ICE_FREE, ROUTES, SearchResult, haversine_km, segments_on_land, turn_deg


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
    def build(cls, land: Land, root: float = 0.25, min_step: float = 1 / 32, lat_limits=ICE_FREE, links=(),
              bathymetry=None, min_depth: float = 15.0) -> "Graph":
        """Given the bathymetry, a cell whose centre is min_depth m deep or less is left out too."""
        x, y, s = split_coast(land, root, min_step, lat_limits)
        lon, lat = x + s / 2, y + s / 2
        water = ~land.contains(lat, lon)
        if bathymetry is not None:
            water &= -bathymetry.elevation(lat, lon) > min_depth
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

    def save(self, path: str, key: str = "") -> None:
        """Keep the graph in an .npz file, with a key saying how it was built."""
        k = [link for *_, link in self.links]
        np.savez(
            path, key=key, lat=self.lat, lon=self.lon, size=self.size, indptr=self.indptr, indices=self.indices, km=self.km,
            link_ends=np.array([(a, b) for a, b, _ in self.links], int).reshape(-1, 2),
            link_names=np.array([link.name for link in k], str), link_points=np.array([len(link.lat) for link in k], int),
            link_lat=np.concatenate([link.lat for link in k] or [[]]), link_lon=np.concatenate([link.lon for link in k] or [[]]),
        )  # fmt: skip

    @classmethod
    def cached(cls, path: str | None, key: str, build) -> "Graph":
        """The graph kept at path if it was built with the same key, else build() kept there."""
        import os

        from shiprouting.sea.links import Link

        if path and os.path.exists(path):
            with np.load(path) as f:
                if str(f["key"]) == key:
                    cut = np.cumsum(f["link_points"])[:-1]
                    links = tuple(
                        (int(a), int(b), Link(str(name), lat, lon))
                        for (a, b), name, lat, lon in zip(f["link_ends"], f["link_names"], np.split(f["link_lat"], cut), np.split(f["link_lon"], cut))
                    )
                    return cls(*(f[n] for n in ("lat", "lon", "size", "indptr", "indices", "km")), links)
        graph = build()
        if path:
            graph.save(path, key)
        return graph


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


def astar(
    graph: Graph, s: int, g: int, edge_cost=None, link_cost=None, heuristic=None, max_seconds: float = math.inf,
    clock0: float = 0.0,
) -> SearchResult:
    """Shortest path from node s to node g.

    The search keeps a clock along the best path to each node, by default the km sailed. By
    default a move costs its length and adds it to the clock. `edge_cost` gives another cost per
    edge, aligned with graph.indices, or is a function `edge_cost(c, clock)` giving the costs of
    node c's edges and how far each moves the clock; `link_cost(link, km, clock)` gives the cost
    and clock step of taking a link. `heuristic(n)` is a lower bound on the cost from n to g.
    The clock starts at clock0.
    The result's km is the length of the path found, cost what was minimised, and clock the
    clock at each point (on a link's own points, spread by distance).
    """
    t0 = time.perf_counter()

    def h(n) -> float:
        return float(haversine_km(graph.lat[n], graph.lon[n], graph.lat[g], graph.lon[g]))

    h = heuristic or h

    jumps = {}  # node -> [(node, km, link, reversed)]
    for a, b, link in graph.links:
        km = (
            float(haversine_km(graph.lat[a], graph.lon[a], link.lat[0], link.lon[0]))
            + link.km
            + float(haversine_km(graph.lat[b], graph.lon[b], link.lat[-1], link.lon[-1]))
        )
        jumps.setdefault(a, []).append((b, km, link, False))
        jumps.setdefault(b, []).append((a, km, link, True))
    edge_cost = graph.km if edge_cost is None else edge_cost
    link_cost = link_cost or (lambda link, km, clock: (km, km))

    best = {s: 0.0}
    clock = {s: clock0}  # along the best path
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
        if time.perf_counter() - t0 > max_seconds:
            raise TimeoutError(f"stopped after {max_seconds:g} s ({expanded} cells expanded)")
        lo, hi = graph.indptr[c], graph.indptr[c + 1]
        w, dt = edge_cost(c, clock[c]) if callable(edge_cost) else (edge_cost[lo:hi], graph.km[lo:hi])
        moves = [(int(n), float(wn), float(tn), None, False) for n, wn, tn in zip(graph.indices[lo:hi], w, dt)]
        moves += [(n, *link_cost(link, km, clock[c]), link, rev) for n, km, link, rev in jumps.get(c, [])]
        for n, wn, tn, link, rev in moves:
            cost = best[c] + wn
            if n not in closed and cost < best.get(n, math.inf):
                best[n] = cost
                clock[n] = clock[c] + tn
                parent[n] = (c, link, rev)
                heapq.heappush(open_list, (cost + h(n), n))
    else:
        raise ValueError(f"no route ({expanded} cells expanded)")

    nodes, via = [g], []  # from goal back to start, each with the link that led into it
    while parent[nodes[-1]] is not None:
        c, link, rev = parent[nodes[-1]]
        via.append((link, rev))
        nodes.append(c)
    lat, lon, at = [graph.lat[s]], [graph.lon[s]], [clock[s]]
    on_link, used = [], []
    for n, (link, rev) in zip(reversed(nodes[:-1]), reversed(via)):
        if link is not None:
            pts_lat, pts_lon = (link.lat[::-1], link.lon[::-1]) if rev else (link.lat, link.lon)
            hop = haversine_km(np.r_[lat[-1], pts_lat], np.r_[lon[-1], pts_lon], np.r_[pts_lat, graph.lat[n]], np.r_[pts_lon, graph.lon[n]])
            share = np.cumsum(hop)[:-1] / hop.sum()
            at += list(at[-1] + share * (clock[n] - at[-1]))
            lat += list(pts_lat)
            lon += list(pts_lon)
            on_link += [False] + [True] * (len(pts_lat) - 1)
            used.append(link.name)
        lat.append(graph.lat[n])
        lon.append(graph.lon[n])
        at.append(clock[n])
        on_link.append(False)
    lat, lon = np.array(lat), np.array(lon)
    km = float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
    return SearchResult(lat, lon, np.array(on_link, bool), used, km, expanded, time.perf_counter() - t0, best[g], np.array(at))


def join_legs(legs: list[SearchResult]) -> SearchResult:
    """The legs as one path, each leg's clock carrying on from where the last one ended."""
    clock = [legs[0].clock]
    for r in legs[1:]:
        clock.append(r.clock[1:] - r.clock[0] + clock[-1][-1])
    return SearchResult(
        np.concatenate([legs[0].lat] + [r.lat[1:] for r in legs[1:]]),
        np.concatenate([legs[0].lon] + [r.lon[1:] for r in legs[1:]]),
        np.concatenate([r.on_link for r in legs]),
        [name for r in legs for name in r.links],
        sum(r.km for r in legs),
        sum(r.expanded for r in legs),
        sum(r.seconds for r in legs),
        sum(r.cost for r in legs),
        np.concatenate(clock),
    )


def straighten(land: Land, r: SearchResult, clearance_km: float = 0.0) -> SearchResult:
    """Cut the corners of a path: from each kept point, go straight to the farthest later point
    the great-circle arc to which stays clearance_km off land. Links are kept as they are."""
    t0 = time.perf_counter()
    lat, lon = r.lat, np.degrees(np.unwrap(np.radians(r.lon)))  # continuous across ±180°
    fixed = np.zeros(len(lat), bool)
    fixed[[0, -1]] = True
    fixed[:-1] |= r.on_link
    fixed[1:] |= r.on_link
    keep, on_link = [0], []
    i = 0
    while i < len(lat) - 1:
        if r.on_link[i]:
            j = i + 1
        else:
            stop = i + 1 + int(np.argmax(fixed[i + 1 :]))
            cand = np.arange(i + 1, stop + 1)
            clear = ~arcs_on_land(land, lat[i], lon[i], lat[cand], lon[cand], clearance_km=clearance_km)
            j = int(cand[clear][-1]) if clear.any() else i + 1
        on_link.append(bool(r.on_link[i]) and j == i + 1)
        keep.append(j)
        i = j
    lat, lon = lat[keep], (lon[keep] + 180) % 360 - 180
    km = float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
    return SearchResult(lat, lon, np.array(on_link, bool), r.links, km, r.expanded, time.perf_counter() - t0)


def smooth_turns(land: Land, r: SearchResult, sharp: float = 60.0, reach: int = 4, keep_points=()) -> SearchResult:
    """Take out the points where the course turns by `sharp` degrees or more.

    The sharpest such point goes first: it and up to reach - 1 neighbours are dropped, the
    fewest that leave a great-circle leg clear of land. Points on links and the two ends
    stay, and so do the points at the indices keep_points (waypoints). Repeats until no sharp
    point can be dropped, then cuts the corners that are left (cut_corners). Unlike straighten, the path stays close to the one found, so a route
    chosen for its weather keeps its way.
    """
    t0 = time.perf_counter()
    lat, lon = r.lat, np.degrees(np.unwrap(np.radians(r.lon)))
    on_link = np.asarray(r.on_link, bool)
    fixed = np.zeros(len(lat), bool)
    fixed[[0, -1]] = True
    fixed[:-1] |= on_link
    fixed[1:] |= on_link
    fixed[list(keep_points)] = True
    keep = np.ones(len(lat), bool)
    stuck = set()
    while True:
        idx = np.nonzero(keep)[0]
        turn = turn_deg(lat[idx], lon[idx])
        order = [k + 1 for k in np.argsort(-turn) if turn[k] >= sharp and not fixed[idx[k + 1]] and idx[k + 1] not in stuck]
        if not order:
            break
        k = order[0]
        for drop in range(1, reach + 1):
            spans = [(a, a + drop) for a in range(k - drop + 1, k + 1) if a >= 1 and a + drop <= len(idx) - 1]
            spans = [(a, b) for a, b in spans if not fixed[idx[a:b]].any()]
            clear = [(a, b) for a, b in spans if not arcs_on_land(land, lat[idx[a - 1]], lon[idx[a - 1]], lat[idx[b]], lon[idx[b]])[0]]
            if clear:
                a, b = clear[0]
                keep[idx[a:b]] = False
                break
        else:
            stuck.add(idx[k])
    idx = np.nonzero(keep)[0]
    seg_link = on_link[idx[:-1]] & (np.diff(idx) == 1)
    clock = None if r.clock is None else r.clock[idx]
    keep_put = np.r_[True, seg_link[:-1] & seg_link[1:], True]  # the two ends and the points inside a link
    keep_put |= np.isin(idx, list(keep_points))
    lat, lon, seg_link, clock = cut_corners(land, lat[idx], lon[idx], keep_put, seg_link, clock, sharp)
    lon = (lon + 180) % 360 - 180
    km = float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
    return SearchResult(lat, lon, seg_link, r.links, km, r.expanded, time.perf_counter() - t0, r.cost, clock)


def cut_corners(land: Land, lat, lon, fixed, seg_link, clock, sharp: float = 60.0, fractions=(0.5, 0.25, 0.1)):
    """Cut off the corners that still turn by `sharp` degrees or more.

    A corner P between A and B gives way to two points the same distance d from P, one
    towards A and one towards B, with d the largest of `fractions` times the shorter leg
    whose new leg is clear of land. The turn is split in two halves, so a corner of 120°
    or more is cut again. Fixed points stay. A link's end may be cut: the new point
    lies on the link's first leg, so the link's way is still followed. Lon is unwrapped
    and stays so. The clock of a new point is read off its leg by distance.
    """
    lat, lon, fixed = np.asarray(lat, float), np.asarray(lon, float), np.asarray(fixed, bool)
    stuck = np.zeros(len(lat), bool)
    for _ in range(4 * len(lat)):
        turn = turn_deg(lat, lon)
        order = [k + 1 for k in np.argsort(-turn) if turn[k] >= sharp and not fixed[k + 1] and not stuck[k + 1]]
        if not order:
            break
        k = order[0]
        legs = haversine_km(lat[k], lon[k], lat[[k - 1, k + 1]], lon[[k - 1, k + 1]])
        d = np.array(fractions) * legs.min()
        fa, fb = d / legs[0], d / legs[1]
        a_lat, a_lon = along(*np.broadcast_arrays(lat[k], lon[k], lat[k - 1], lon[k - 1], fa))
        b_lat, b_lon = along(*np.broadcast_arrays(lat[k], lon[k], lat[k + 1], lon[k + 1], fb))
        a_lon, b_lon = (lon[k] + (v - lon[k] + 180) % 360 - 180 for v in (a_lon, b_lon))
        clear = ~arcs_on_land(land, a_lat, a_lon, b_lat, b_lon)
        if not clear.any():
            stuck[k] = True
            continue
        j = int(np.argmax(clear))
        lat = np.r_[lat[:k], a_lat[j], b_lat[j], lat[k + 1:]]
        lon = np.r_[lon[:k], a_lon[j], b_lon[j], lon[k + 1:]]
        fixed, stuck = (np.r_[v[:k], False, False, v[k + 1:]] for v in (fixed, stuck))
        seg_link = np.r_[seg_link[:k], False, seg_link[k:]]
        if clock is not None:
            ca, cb = (clock[k] + f[j] * (clock[n] - clock[k]) for f, n in ((fa, k - 1), (fb, k + 1)))
            clock = np.r_[clock[:k], ca, cb, clock[k + 1:]]
    return lat, lon, seg_link, clock

def arcs_on_land(land: Land, lat0, lon0, lat1, lon1, piece_km: float = 20.0, clearance_km: float = 0.0) -> np.ndarray:
    """Whether each great-circle arc from (lat0, lon0) to (lat1, lon1) comes within clearance_km of land.

    An arc is cut into pieces of at most piece_km, each tested as a straight lon/lat line;
    over 20 km the two differ by metres. The clearance is turned into degrees with the
    longitude scale at the piece, so it is at least clearance_km in every direction.
    """
    lat0, lon0, lat1, lon1 = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in (lat0, lon0, lat1, lon1)))
    n = np.maximum(1, np.ceil(haversine_km(lat0, lon0, lat1, lon1) / piece_km)).astype(int)
    arc = np.repeat(np.arange(len(n)), n)
    k = np.arange(len(arc)) - np.repeat(np.cumsum(n) - n, n)
    a_lat, a_lon = along(lat0[arc], lon0[arc], lat1[arc], lon1[arc], k / n[arc])
    b_lat, b_lon = along(lat0[arc], lon0[arc], lat1[arc], lon1[arc], (k + 1) / n[arc])
    b_lon = a_lon + (b_lon - a_lon + 180) % 360 - 180  # a piece crossing ±180° runs past it
    a, b = np.column_stack([a_lon, a_lat]), np.column_stack([b_lon, b_lat])
    deg = clearance_km / (111.32 * np.cos(np.radians((a_lat + b_lat) / 2)))

    def touches(lines, k):
        return land.crosses(lines) if clearance_km <= 0 else land.near(lines, deg[k])

    hit = touches(shapely.linestrings(np.stack([a, b], axis=1)), slice(None))
    for shift, past in ((360.0, b_lon < -180), (-360.0, b_lon > 180)):
        if past.any():
            hit[past] |= touches(shapely.linestrings(np.stack([a[past] + [shift, 0], b[past] + [shift, 0]], axis=1)), past)
    return np.bincount(arc, weights=hit, minlength=len(n)) > 0


def along(lat0, lon0, lat1, lon1, f):
    """The point a fraction f of the way along the great circle from (lat0, lon0) to (lat1, lon1)."""
    p0, l0, p1, l1 = (np.radians(v) for v in (lat0, lon0, lat1, lon1))
    u0 = np.stack([np.cos(p0) * np.cos(l0), np.cos(p0) * np.sin(l0), np.sin(p0)])
    u1 = np.stack([np.cos(p1) * np.cos(l1), np.cos(p1) * np.sin(l1), np.sin(p1)])
    omega = np.arccos(np.clip((u0 * u1).sum(axis=0), -1.0, 1.0))
    sin_omega = np.where(omega > 1e-12, np.sin(omega), 1.0)
    w0 = np.where(omega > 1e-12, np.sin((1 - f) * omega) / sin_omega, 1 - f)
    w1 = np.where(omega > 1e-12, np.sin(f * omega) / sin_omega, f)
    u = w0 * u0 + w1 * u1
    return np.degrees(np.arctan2(u[2], np.hypot(u[0], u[1]))), np.degrees(np.arctan2(u[1], u[0]))


def run(land: Land, min_step: float, links=(), clearance_km: float = 0.0) -> list[dict]:
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
            "clearance_km": clearance_km,
            "snap_km": round(float(max(haversine_km(*start, graph.lat[s], graph.lon[s]), haversine_km(*goal, graph.lat[g], graph.lon[g]))), 1),
            "great_circle_km": round(float(haversine_km(*start, *goal))),
        }
        try:
            r = astar(graph, s, g)
        except ValueError as e:
            row["error"] = str(e)
        else:
            near = [n for n, (la, lo) in CHECKPOINTS.items() if haversine_km(r.lat, r.lon, la, lo).min() < 50]
            t = straighten(land, r, clearance_km)
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
                straight_km=round(t.km),
                straight_seconds=round(t.seconds, 1),
                straight_legs=len(t.lat) - 1,
                straight_legs_touching_land=int((arcs_on_land(land, t.lat[:-1], t.lon[:-1], t.lat[1:], t.lon[1:]) & ~t.on_link).sum()),
                straight_lat=np.round(t.lat, 4).tolist(),
                straight_lon=np.round(t.lon, 4).tolist(),
            )
        print({k: v for k, v in row.items() if k not in ("lat", "lon", "straight_lat", "straight_lon")}, flush=True)
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
    parser.add_argument("--clearance-km", type=float, default=0.0, help="how far straightened legs keep off land")
    args = parser.parse_args()
    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    rows = [row for step in args.min_step or [1 / 32] for row in run(land, step, links, args.clearance_km)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
