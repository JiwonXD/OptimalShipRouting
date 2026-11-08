"""The least fuel arriving by a deadline, searched with time in the state: a check on meet_eta.

meet_eta puts a price on an hour and bisects it (Lagrangian relaxation). That finds only routes
on the lower convex hull of fuel against time, so it can miss the least-fuel route that just
meets the deadline: the duality gap. Here time is part of the state instead. A label is a
cell, an arrival time and the fuel burnt to get there; labels are taken in order of time,
each move at every speed. At each cell, labels whose times fall in the same bucket of
dt_hours are compared, and one is dropped if another arrives no later and burns no more
fuel. Labels in different buckets are all kept, so a later arrival that meets better
weather is not lost; within a bucket, the times differ by less than dt_hours, which is the
only approximation. A label is dropped when it cannot reach the goal by the deadline even
at the fastest speed along the great circle with the strongest current behind it.

This is far slower than meet_eta and is meant for small areas (a few thousand cells), or a
corridor around a path (allowed). Links are not taken.
"""

import heapq
import math
import time

import numpy as np

from shiprouting.route.speed import SpeedChoice
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.quadtree import Graph


def least_fuel_by(graph: Graph, s: int, g: int, choice: SpeedChoice, eta_hours: float, dt_hours: float = 0.1,
                  max_current: float = 2.5, max_labels: int = 5_000_000, max_seconds: float = math.inf,
                  upper_kg: float = math.inf, allowed: np.ndarray | None = None) -> SearchResult:
    """The least-fuel path from node s to node g arriving within eta_hours, each move at one of
    choice's speeds. max_current (m/s) bounds the current that can help, for pruning. upper_kg:
    the fuel of a path known to arrive in time (meet_eta's, say); labels burning that much are
    dropped from the start, and if none does better a ValueError says so. The result's cost is
    the fuel, kg, and its clock the hours at each point. allowed: a bool per node; the search
    keeps to the nodes marked (a corridor around a known path, say), so it is exact within them."""
    t0 = time.perf_counter()
    fastest_kmh = (max(c.speed for c in choice.costs) + max_current) * 3.6
    to_goal_h = haversine_km(graph.lat, graph.lon, graph.lat[g], graph.lon[g]) / fastest_kmh
    if allowed is not None:
        to_goal_h = np.where(allowed, to_goal_h, math.inf)
    labels_t, labels_f, labels_node, labels_parent = [0.0], [0.0], [s], [-1]
    kept = {(s, 0): [0]}  # (node, bucket) -> label ids still standing
    heap = [(0.0, 0)]
    dead = set()
    goal = None

    def bound():
        return labels_f[goal] if goal is not None else upper_kg

    while heap:
        t, lab = heapq.heappop(heap)
        if lab in dead:
            continue
        c, f = labels_node[lab], labels_f[lab]
        if c == g:
            if goal is None or f < labels_f[goal]:
                goal = lab
            continue
        if f >= bound():
            continue  # fuel only grows
        if len(labels_t) > max_labels or time.perf_counter() - t0 > max_seconds:
            raise TimeoutError(f"stopped with {len(labels_t)} labels after {time.perf_counter() - t0:.0f} s")
        lo, hi = graph.indptr[c], graph.indptr[c + 1]
        n = graph.indices[lo:hi]
        kg, h = choice.kg_hours(graph.lat[c], graph.lon[c], graph.lat[n], graph.lon[n], t)  # (speeds, moves)
        for k in range(kg.shape[0]):
            for m in range(kg.shape[1]):
                kg_m, h_m = kg[k, m], h[k, m]
                if not math.isfinite(kg_m):
                    continue
                nn, tn, fn = int(n[m]), t + float(h_m), f + float(kg_m)
                if tn + to_goal_h[nn] > eta_hours * (1 + 1e-9):
                    continue
                if fn >= bound():
                    continue
                key = (nn, int(tn // dt_hours))
                others = kept.get(key, [])
                if any(labels_t[o] <= tn and labels_f[o] <= fn for o in others):
                    continue
                for o in others:
                    if tn <= labels_t[o] and fn <= labels_f[o]:
                        dead.add(o)
                new = len(labels_t)
                labels_t.append(tn)
                labels_f.append(fn)
                labels_node.append(nn)
                labels_parent.append(lab)
                kept[key] = [o for o in others if o not in dead] + [new]
                heapq.heappush(heap, (tn, new))
    if goal is None:
        raise ValueError(f"no path arrives within {eta_hours:g} h burning less than {upper_kg:g} kg ({len(labels_t)} labels)")
    path = [goal]
    while labels_parent[path[-1]] >= 0:
        path.append(labels_parent[path[-1]])
    path.reverse()
    nodes = np.array([labels_node[p] for p in path])
    lat, lon = graph.lat[nodes], graph.lon[nodes]
    km = float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
    return SearchResult(lat, lon, np.zeros(len(nodes) - 1, bool), [], km, len(labels_t), time.perf_counter() - t0,
                        labels_f[goal], np.array([labels_t[p] for p in path]))
