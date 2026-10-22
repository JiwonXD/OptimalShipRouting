"""Shortest sea route on the graph, then straightened against the land."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from shiprouting.sea.geometry import great_circle_points, haversine_km, path_geometry
from shiprouting.sea.grid import SeaGraph
from shiprouting.sea.land import Land


def _unit(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)])


class Router:
    def __init__(self, graph: SeaGraph, land: Land):
        self.graph = graph
        self.land = land
        n = len(graph.cells)
        w = np.concatenate([graph.edge_km, graph.edge_km])
        self.matrix = coo_matrix(
            (w, (np.concatenate([graph.edge_i, graph.edge_j]), np.concatenate([graph.edge_j, graph.edge_i]))),
            shape=(n, n),
        ).tocsr()
        self.tree = cKDTree(_unit(graph.lat, graph.lon))

    def nearest_node(self, lat: float, lon: float) -> int:
        return int(self.tree.query(_unit([lat], [lon])[0])[1])

    def shortest_nodes(self, start: int, end: int) -> list[int]:
        """Node indices of the shortest path (Dijkstra, exact on the graph)."""
        dist, pred = dijkstra(self.matrix, indices=start, return_predecessors=True)
        if not np.isfinite(dist[end]):
            raise ValueError("no sea route between these points")
        path = [end]
        while path[-1] != start:
            path.append(int(pred[path[-1]]))
        return path[::-1]

    def clear(self, lat1, lon1, lat2, lon2, step_km: float = 2.0) -> bool:
        """True if the great circle between the points stays off land."""
        lat, lon = great_circle_points(lat1, lon1, lat2, lon2, step_km)
        return not self.land.crosses([path_geometry(lat, lon)])[0]

    def straighten(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Drop waypoints while the great circle that replaces them stays off land."""
        keep = [0]
        k = 1
        while k < len(lat) - 1:
            a = keep[-1]
            if self.clear(lat[a], lon[a], lat[k + 1], lon[k + 1]):
                k += 1
            else:
                keep.append(k)
                k += 1
        keep.append(len(lat) - 1)
        return np.asarray(lat)[keep], np.asarray(lon)[keep]

    def route(self, start: tuple[float, float], end: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Waypoints (lat, lon) from ``start`` to ``end``, each a (lat, lon) pair at sea."""
        nodes = self.shortest_nodes(self.nearest_node(*start), self.nearest_node(*end))
        lat = np.concatenate([[start[0]], self.graph.lat[nodes], [end[0]]])
        lon = np.concatenate([[start[1]], self.graph.lon[nodes], [end[1]]])
        return self.straighten(lat, lon)


def length_km(lat, lon) -> float:
    return float(np.sum(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])))
