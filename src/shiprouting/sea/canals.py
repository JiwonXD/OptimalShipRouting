"""Canals as explicit edges of the sea graph.

A canal is too narrow for the grid cells, so it is taken from the OSM
waterway centrelines (an Overpass JSON answer with ``out geom``) as a
polyline between two given ends.
"""

import json
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from shiprouting.sea.geometry import haversine_km

# Approximate ends at sea, snapped to the nearest node of the centrelines.
ENDS = {
    "Suez": ((31.276, 32.326), (29.930, 32.561)),  # Port Said, Suez
    "Panama": ((8.943, -79.565), (9.311, -79.920)),  # Pacific, Atlantic
}


@dataclass(frozen=True)
class Canal:
    name: str
    lat: np.ndarray  # centreline from one end to the other, degrees
    lon: np.ndarray

    @property
    def length_km(self) -> float:
        return float(np.sum(haversine_km(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])))


def from_overpass(name: str, path, ends=None) -> Canal:
    """Shortest path along the waterway ways between the two ``ends``."""
    with open(path, encoding="utf-8") as f:
        ways = [e for e in json.load(f)["elements"] if e["type"] == "way"]
    ids, coords = {}, []
    rows, cols = [], []
    for way in ways:
        prev = None
        for node, point in zip(way["nodes"], way["geometry"]):
            if node not in ids:
                ids[node] = len(coords)
                coords.append((point["lat"], point["lon"]))
            k = ids[node]
            if prev is not None:
                rows.append(prev)
                cols.append(k)
            prev = k
    lat, lon = np.array(coords).T
    rows, cols = np.array(rows), np.array(cols)
    w = haversine_km(lat[rows], lon[rows], lat[cols], lon[cols])
    n = len(coords)
    graph = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), (n, n))

    (lat_a, lon_a), (lat_b, lon_b) = ends or ENDS[name]
    a = int(np.argmin(haversine_km(lat, lon, lat_a, lon_a)))
    b = int(np.argmin(haversine_km(lat, lon, lat_b, lon_b)))
    dist, pred = dijkstra(graph.tocsr(), indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        raise ValueError(f"{name}: the two ends are not connected")
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    path = path[::-1]
    return Canal(name, lat[path], lon[path])
