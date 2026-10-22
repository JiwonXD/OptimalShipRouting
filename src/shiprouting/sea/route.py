"""Shortest sea route on the graph, then straightened against the land."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from shiprouting.sea.canals import Canal
from shiprouting.sea.geometry import great_circle_points, haversine_km, path_geometry
from shiprouting.sea.grid import SeaGraph
from shiprouting.sea.land import Land


def _unit(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)])


class Router:
    def __init__(self, graph: SeaGraph, land: Land, canals: tuple[Canal, ...] = ()):
        self.graph = graph
        self.land = land
        self.nodes = np.arange(len(graph.cells))
        self.tree = cKDTree(_unit(graph.lat, graph.lon))
        # A canal is one edge between the sea nodes nearest its two ends.
        self.canals = {}
        i, j, km = [graph.edge_i], [graph.edge_j], [graph.edge_km]
        for canal in canals:
            a = self.nearest_node(canal.lat[0], canal.lon[0])
            b = self.nearest_node(canal.lat[-1], canal.lon[-1])
            ends = haversine_km(graph.lat[[a, b]], graph.lon[[a, b]], canal.lat[[0, -1]], canal.lon[[0, -1]])
            self.canals[a, b] = (canal.lat, canal.lon)
            self.canals[b, a] = (canal.lat[::-1], canal.lon[::-1])
            i.append([a])
            j.append([b])
            km.append([canal.length_km + ends.sum()])
        i, j, km = np.concatenate(i), np.concatenate(j), np.concatenate(km)
        n = len(graph.cells)
        self.matrix = coo_matrix(
            (np.concatenate([km, km]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n)
        ).tocsr()
        # Snap points only to the largest connected sea, not to lakes and enclosed inlets.
        _, labels = connected_components(self.matrix, directed=False)
        self.nodes = np.flatnonzero(labels == np.bincount(labels).argmax())
        self.tree = cKDTree(_unit(graph.lat[self.nodes], graph.lon[self.nodes]))

    def nearest_node(self, lat: float, lon: float) -> int:
        return int(self.nodes[self.tree.query(_unit([lat], [lon])[0])[1]])

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
        lat, lon = [[start[0]]], [[start[1]]]
        for k, u in enumerate(nodes):
            lat.append([self.graph.lat[u]])
            lon.append([self.graph.lon[u]])
            if k + 1 < len(nodes) and (u, nodes[k + 1]) in self.canals:
                c_lat, c_lon = self.canals[u, nodes[k + 1]]
                lat.append(c_lat)
                lon.append(c_lon)
        lat.append([end[0]])
        lon.append([end[1]])
        return self.straighten(np.concatenate(lat), np.concatenate(lon))


def length_km(lat, lon) -> float:
    return float(np.sum(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])))


def main() -> None:
    import argparse
    import json

    from shiprouting.sea.canals import from_overpass
    from shiprouting.sea.land import load_osm_land_polygons

    parser = argparse.ArgumentParser(description="Shortest sea route between two points.")
    parser.add_argument("graph", help="sea graph .npz")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("start", help="lat,lon")
    parser.add_argument("end", help="lat,lon")
    parser.add_argument("--canal", action="append", default=[], help="NAME=overpass.json (NAME in canals.ENDS)")
    parser.add_argument("--geojson", help="write the route as GeoJSON")
    args = parser.parse_args()

    canals = tuple(from_overpass(*c.split("=", 1)) for c in args.canal)
    router = Router(SeaGraph.load(args.graph), load_osm_land_polygons(args.land), canals)
    start, end = (tuple(float(x) for x in p.split(",")) for p in (args.start, args.end))
    lat, lon = router.route(start, end)
    print(f"{length_km(lat, lon):.0f} km, {len(lat)} waypoints")
    if args.geojson:
        line = {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in zip(lon, lat)]}
        with open(args.geojson, "w") as f:
            json.dump({"type": "Feature", "geometry": line, "properties": {"km": length_km(lat, lon)}}, f)


if __name__ == "__main__":
    main()
