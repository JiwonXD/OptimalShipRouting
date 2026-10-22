"""Spherical geometry on longitude/latitude degrees, safe across the dateline."""

import numpy as np
import shapely

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius
_WORLD = shapely.box(-180.0, -90.0, 180.0, 90.0)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance, km (arrays broadcast)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _unit(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)], axis=-1)


def great_circle_points(lat1, lon1, lat2, lon2, step_km):
    """Points along the great circle, at most ``step_km`` apart, as (lat, lon) arrays."""
    a, b = _unit(lat1, lon1), _unit(lat2, lon2)
    omega = np.arccos(np.clip(a @ b, -1.0, 1.0))
    n = max(1, int(np.ceil(omega * EARTH_RADIUS_KM / step_km)))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    if omega < 1e-12:
        v = np.repeat(a[None, :], n + 1, axis=0)
    else:
        v = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
    lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return lat, lon


def path_geometry(lat, lon):
    """Line through the points; a line crossing the dateline is drawn on both sides of it."""
    lon = np.degrees(np.unwrap(np.radians(lon)))
    line = shapely.linestrings(np.column_stack([lon, lat]))
    if lon.min() >= -180.0 and lon.max() <= 180.0:
        return line
    shift = 360.0 if lon.min() < -180.0 else -360.0
    return shapely.multilinestrings([line, shapely.transform(line, lambda c: c + [shift, 0.0])])


def ring_geometry(lat, lon):
    """Polygon from a closed boundary (not containing a pole), clipped to the lon/lat world box."""
    lon = np.asarray(lon, float)
    if lon.max() - lon.min() <= 180.0:
        return shapely.polygons(np.column_stack([lon, lat]))
    lon = np.where(lon < 0, lon + 360.0, lon)
    poly = shapely.polygons(np.column_stack([lon, lat]))
    both = shapely.multipolygons([poly, shapely.transform(poly, lambda c: c - [360.0, 0.0])])
    return shapely.intersection(both, _WORLD)
