"""Water depth from the GEBCO grid, and where routes run over shallow water.

GEBCO gives elevation (m, negative below sea level) on a 15 arc-second grid,
pixel-centre registered, as one int16 array in a netCDF-4 (HDF5) file. The
array is stored contiguously, so it is mapped from disk rather than read: the
7.5 GB file costs no memory, and only the pages touched are loaded.
"""

import numpy as np
import shapely

from shiprouting.sea.latlon import haversine_km
from shiprouting.sea.quadtree import along


class Bathymetry:
    def __init__(self, path):
        import h5py

        with h5py.File(path, "r") as f:
            z = f["elevation"]
            offset = z.id.get_offset()
            if offset is None:
                raise ValueError(f"{path}: elevation is not stored contiguously")
            self.lat0, self.lon0 = float(f["lat"][0]), float(f["lon"][0])
            self.step = float(f["lat"][1] - f["lat"][0])
            shape, dtype = z.shape, z.dtype
        self.z = np.memmap(path, dtype, "r", offset, shape)

    def elevation(self, lat, lon) -> np.ndarray:
        """Elevation, m, of the grid cell holding each point (negative below sea level)."""
        i = np.floor((np.asarray(lat, float) - self.lat0) / self.step + 0.5).astype(int)
        j = np.floor((np.asarray(lon, float) - self.lon0) / self.step + 0.5).astype(int) % self.z.shape[1]
        return np.asarray(self.z[np.clip(i, 0, self.z.shape[0] - 1), j])


class LandOrShallow:
    """Land, and water min_depth m deep or less, behind the queries Land answers.

    A line is checked against the depth every quarter grid step along it (a straight
    lon/lat line, as Land checks it). Within exempt_km of an exempt point (the sea
    ends of the links) depth is not checked: the dredged channels into canals are
    narrower than the grid. Splitting the coast into small cells still follows the
    land alone (tree, fraction).
    """

    def __init__(self, land, bathy: Bathymetry, min_depth: float = 15.0, exempt=(), exempt_km: float = 0.0,
                 chunk: int = 2_000_000):
        self.land, self.bathy, self.min_depth, self.chunk = land, bathy, min_depth, chunk
        self.exempt, self.exempt_km = list(exempt), exempt_km
        self.tree, self.fraction = land.tree, land.fraction

    def shallow(self, lat, lon) -> np.ndarray:
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        out = np.array(-self.bathy.elevation(lat, lon) <= self.min_depth)
        out[out] = ~exempted(lat[out], lon[out], self.exempt, self.exempt_km)  # only where shallow: most samples are deep
        return out

    def contains(self, lat, lon) -> np.ndarray:
        return self.land.contains(lat, lon) | self.shallow(lat, lon)

    def crosses(self, lines) -> np.ndarray:
        return self.land.crosses(lines) | self.over_shallow(lines)

    def near(self, geoms, distance) -> np.ndarray:
        return self.land.near(geoms, distance) | self.over_shallow(geoms)

    def over_shallow(self, lines) -> np.ndarray:
        """Whether each two-point line passes over shallow water."""
        lines = np.asarray(lines, dtype=object)
        xy = shapely.get_coordinates(lines).reshape(-1, 2, 2)
        n = np.ceil(np.abs(xy[:, 1] - xy[:, 0]).max(axis=1) / (self.bathy.step / 4)).astype(int) + 1
        hit = np.zeros(len(n), bool)
        first = 0
        while first < len(n):  # lines in chunks of about self.chunk samples
            last = first + max(1, int(np.searchsorted(np.cumsum(n[first:]), self.chunk)))
            m = n[first:last]
            line = np.repeat(np.arange(first, last), m)
            f = (np.arange(len(line)) - np.repeat(np.cumsum(m) - m, m)) / np.repeat(np.maximum(m - 1, 1), m)
            p = xy[line, 0] + f[:, None] * (xy[line, 1] - xy[line, 0])
            hit[first:last] = np.bincount(line - first, weights=self.shallow(p[:, 1], p[:, 0]), minlength=last - first) > 0
            first = last
        return hit.reshape(lines.shape)


def exempted(lat, lon, points, km: float) -> np.ndarray:
    """Whether each point lies within km of any of the points (lat, lon)."""
    out = np.zeros(np.shape(lat), bool)
    for p_lat, p_lon in points if km > 0 else ():
        out |= haversine_km(lat, lon, p_lat, p_lon) <= km
    return out


def link_ends():
    """The sea ends of the canal and strait links, (lat, lon)."""
    from shiprouting.sea.links import LINKS

    return [end for _, a, b, _ in LINKS.values() for end in (a, b)]


# Straits whose water deeper than the limit is narrower than the grid's cells, so a cell over
# them is shallow at its centre and left out: (sea end, sea end), each in water the grid has.
CHANNELS = {
    "그레이트벨트": ((56.6, 11.6), (54.6, 11.3)),  # Kattegat to the Fehmarn Belt, the Baltic's deep-water way
}


def channel(bathy, land, name: str, a, b, min_depth: float = 15.0, pad: float = 1.0):
    """A link from sea end a to b over water deeper than min_depth: the shortest path over the
    GEBCO cells of the box around the two ends (and pad degrees more) that are deep and off land,
    then cut to the fewest corners whose straight legs stay over such cells."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    from shiprouting.sea.links import Link

    lo, hi = np.minimum(a, b) - pad, np.maximum(a, b) + pad
    rows = np.arange(*np.round((np.array([lo[0], hi[0]]) - bathy.lat0) / bathy.step).astype(int))
    cols = np.arange(*np.round((np.array([lo[1], hi[1]]) - bathy.lon0) / bathy.step).astype(int))
    lat, lon = np.meshgrid(bathy.lat0 + rows * bathy.step, bathy.lon0 + cols * bathy.step, indexing="ij")

    def deep(p_lat, p_lon):
        p_lat, p_lon = np.ravel(p_lat), np.ravel(p_lon)
        return (-bathy.elevation(p_lat, p_lon) > min_depth) & ~land.contains(p_lat, p_lon)

    ok = deep(lat, lon).reshape(lat.shape)
    n, m = ok.shape
    k = np.arange(n * m).reshape(n, m)
    i, j = [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        x = ok[: n - di, max(0, -dj): m - max(0, dj)] & ok[di:, max(0, dj): m - max(0, -dj)]
        x &= ok[: n - di, max(0, dj): m - max(0, -dj)] & ok[di:, max(0, -dj): m - max(0, dj)]  # no corner cut over a shallow cell
        i.append(k[: n - di, max(0, -dj): m - max(0, dj)][x])
        j.append(k[di:, max(0, dj): m - max(0, -dj)][x])
    i, j = np.concatenate(i), np.concatenate(j)
    flat_lat, flat_lon = lat.ravel(), lon.ravel()
    km = haversine_km(flat_lat[i], flat_lon[i], flat_lat[j], flat_lon[j])
    s, g = (int(np.argmin(np.where(ok.ravel(), haversine_km(flat_lat, flat_lon, *p), np.inf))) for p in (a, b))
    dist, parent = dijkstra(coo_matrix((km, (i, j)), shape=(n * m, n * m)).tocsr(), directed=False, indices=s,
                            return_predecessors=True)
    if not np.isfinite(dist[g]):
        raise ValueError(f"{name}: no water deeper than {min_depth:g} m joins the two ends")
    path = [g]
    while path[-1] != s:
        path.append(int(parent[path[-1]]))
    p_lat, p_lon = flat_lat[path[::-1]], flat_lon[path[::-1]]

    def clear(u, v):  # the straight leg, sampled every quarter cell
        t = np.linspace(0, 1, int(max(abs(p_lat[v] - p_lat[u]), abs(p_lon[v] - p_lon[u])) / (bathy.step / 4)) + 2)
        return deep(p_lat[u] + t * (p_lat[v] - p_lat[u]), p_lon[u] + t * (p_lon[v] - p_lon[u])).all()

    keep = [0]
    while keep[-1] < len(path) - 1:  # the farthest point still in a clear straight line (the next one always is)
        u, v = keep[-1], keep[-1] + 1
        while v + 1 < len(path) and clear(u, v + 1):
            v += 1
        keep.append(v)
    return Link(name, p_lat[keep], p_lon[keep])


def shallowest(bathy: Bathymetry, lat, lon, piece_km: float = 0.25, exempt=(), exempt_km: float = 0.0):
    """Per leg of a path: the least depth along it, m, sampled every piece_km on the great circle, and where.
    Points within exempt_km of an exempt point are not counted."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n = np.maximum(1, np.ceil(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / piece_km)).astype(int) + 1
    leg = np.repeat(np.arange(len(n)), n)
    f = (np.arange(len(leg)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    p_lat, p_lon = along(lat[leg], lon[leg], lat[leg + 1], lon[leg + 1], f)
    depth = np.where(exempted(p_lat, p_lon, exempt, exempt_km), np.inf, -bathy.elevation(p_lat, p_lon).astype(float))
    start = np.cumsum(n) - n
    least = np.minimum.reduceat(depth, start)
    at = start + np.array([np.argmin(depth[a:a + k]) for a, k in zip(start, n)])
    return least, p_lat[at], p_lon[at]


def summary(bathy: Bathymetry, lat, lon, on_link=None, min_depth: float = 15.0, exempt=(), exempt_km: float = 0.0) -> dict:
    """Legs that run over water min_depth m deep or less (links and exempt zones aside), and the depth at the two ends."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    on_link = np.zeros(len(lat) - 1, bool) if on_link is None else np.asarray(on_link, bool)
    least, w_lat, w_lon = shallowest(bathy, lat, lon, exempt=exempt, exempt_km=exempt_km)
    shallow = np.nonzero((least <= min_depth) & ~on_link)[0]
    ends = -bathy.elevation(lat[[0, -1]], lon[[0, -1]]).astype(float)
    return {
        "legs": int((~on_link).sum()),
        "shallow_legs": len(shallow),
        "least_m": round(float(least[~on_link].min()), 1) if np.isfinite(least[~on_link]).any() else None,
        "start_m": float(ends[0]),
        "goal_m": float(ends[1]),
        "where": [(round(float(w_lat[k]), 3), round(float(w_lon[k]), 3), float(least[k])) for k in shallow],
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Legs of found routes that run over shallow water.")
    parser.add_argument("gebco", help="GEBCO grid, netCDF")
    parser.add_argument("results", nargs="+", help="results .json files with lat/lon (and on_link) per route")
    parser.add_argument("--min-depth", type=float, default=15.0, help="water this deep or less counts as shallow, m")
    parser.add_argument("--link-approach-km", type=float, default=0.0, help="leave out water this close to a link's sea end")
    args = parser.parse_args()

    bathy = Bathymetry(args.gebco)
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        rows = [rows] if isinstance(rows, dict) else rows  # one plan, or rows of test voyages
        print(path)
        for row in rows:
            for key, found in row.items():
                if isinstance(found, dict) and "lat" in found:
                    s = summary(bathy, found["lat"], found["lon"], found.get("on_link"), args.min_depth, link_ends(), args.link_approach_km)
                    print(f"  {row.get('route', '')} {key}: {s['shallow_legs']} of {s['legs']} legs ≤{args.min_depth:g} m "
                          f"(least {s['least_m']} m, start {s['start_m']:g} m, goal {s['goal_m']:g} m) {s['where'][:10]}")


if __name__ == "__main__":
    main()
