"""Land polygons and the queries the sea grid needs.

The intended source is the OpenStreetMap land polygons, split version in
WGS84 (land-polygons-split-4326, © OpenStreetMap contributors, ODbL). Inland
lakes are part of the land there, so only water connected to the sea remains.
"""

import numpy as np
import shapely


class Land:
    def __init__(self, polygons):
        self.geoms = np.asarray(polygons, dtype=object)
        self.tree = shapely.STRtree(self.geoms)

    def fraction(self, polys) -> np.ndarray:
        """Share of each polygon's lon/lat area that is land (polygons must not overlap)."""
        polys = np.asarray(polys, dtype=object)
        i, j = self.tree.query(polys, predicate="intersects")
        inter = shapely.area(shapely.intersection(polys[i], self.geoms[j]))
        land_area = np.bincount(i, weights=inter, minlength=len(polys))
        return land_area / shapely.area(polys)

    def contains(self, lat, lon) -> np.ndarray:
        """True where the point is on land (or on the coastline)."""
        points = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
        i, _ = self.tree.query(points, predicate="intersects")
        hit = np.zeros(points.shape, bool)
        hit[i] = True
        return hit

    def crosses(self, lines) -> np.ndarray:
        """True where the line touches land."""
        lines = np.asarray(lines, dtype=object)
        i, _ = self.tree.query(lines, predicate="intersects")
        hit = np.zeros(lines.shape, bool)
        hit[i] = True
        return hit


def load_osm_land_polygons(path) -> Land:
    """Read the OSM land polygons shapefile (or the zip it comes in)."""
    import pyogrio

    _, _, wkb, _ = pyogrio.raw.read(path, read_geometry=True, columns=[])
    return Land(shapely.from_wkb(wkb))
