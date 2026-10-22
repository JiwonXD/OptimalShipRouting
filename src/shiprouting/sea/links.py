"""Waterways too narrow for the grid, added as explicit links between two sea ends.

A link is a polyline the ship follows from end to end; its cost is its length.
The geometry comes from OpenStreetMap ways (Overpass JSON with `out geom`),
chained through shared nodes from the node nearest one sea end to the node
nearest the other.
"""

import heapq
import json
from dataclasses import dataclass

import numpy as np

from shiprouting.sea.latlon import haversine_km

# Sea ends (lat, lon) of each canal, and the Overpass file holding its ways.
CANALS = {
    "수에즈": ("canals_suez.json", (31.2757, 32.3263), (29.9303, 32.5607)),
    "파나마": ("canals_panama.json", (8.9429, -79.5647), (9.3115, -79.9198)),
}


@dataclass(frozen=True)
class Link:
    name: str
    lat: np.ndarray  # polyline from one end to the other
    lon: np.ndarray

    @property
    def km(self) -> float:
        return float(haversine_km(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:]).sum())


def chain(name: str, ways: list[dict], a: tuple[float, float], b: tuple[float, float]) -> Link:
    """Shortest path over the ways' nodes from the node nearest `a` to the node nearest `b`."""
    pos, edges = {}, {}
    for way in ways:
        nodes = way["nodes"]
        for n, g in zip(nodes, way["geometry"]):
            pos[n] = (g["lat"], g["lon"])
        for u, v in zip(nodes[:-1], nodes[1:]):
            km = float(haversine_km(*pos[u], *pos[v]))
            edges.setdefault(u, []).append((v, km))
            edges.setdefault(v, []).append((u, km))
    ids = list(pos)
    lat, lon = np.array([pos[n] for n in ids]).T
    s = ids[int(np.argmin(haversine_km(lat, lon, *a)))]
    g = ids[int(np.argmin(haversine_km(lat, lon, *b)))]

    best, parent, open_list = {s: 0.0}, {s: None}, [(0.0, s)]
    while open_list:
        d, u = heapq.heappop(open_list)
        if u == g:
            break
        if d > best[u]:
            continue
        for v, km in edges.get(u, ()):
            if d + km < best.get(v, np.inf):
                best[v], parent[v] = d + km, u
                heapq.heappush(open_list, (d + km, v))
    else:
        raise ValueError(f"{name}: the ways do not connect the two ends")

    path = [g]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    lat, lon = np.array([pos[n] for n in reversed(path)]).T
    return Link(name, lat, lon)


def load_canals(osm_dir) -> list[Link]:
    from pathlib import Path

    links = []
    for name, (file, a, b) in CANALS.items():
        data = json.loads((Path(osm_dir) / file).read_text(encoding="utf-8"))
        links.append(chain(name, [e for e in data["elements"] if e["type"] == "way"], a, b))
    return links
