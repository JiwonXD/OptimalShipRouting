"""Water depth from the GEBCO grid, and where routes run over shallow water.

GEBCO gives elevation (m, negative below sea level) on a 15 arc-second grid,
pixel-centre registered, as one int16 array in a netCDF-4 (HDF5) file. The
array is stored contiguously, so it is mapped from disk rather than read: the
7.5 GB file costs no memory, and only the pages touched are loaded.
"""

import numpy as np

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


def shallowest(bathy: Bathymetry, lat, lon, piece_km: float = 0.25):
    """Per leg of a path: the least depth along it, m, sampled every piece_km on the great circle, and where."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n = np.maximum(1, np.ceil(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / piece_km)).astype(int) + 1
    leg = np.repeat(np.arange(len(n)), n)
    f = (np.arange(len(leg)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    p_lat, p_lon = along(lat[leg], lon[leg], lat[leg + 1], lon[leg + 1], f)
    depth = -bathy.elevation(p_lat, p_lon).astype(float)
    start = np.cumsum(n) - n
    least = np.minimum.reduceat(depth, start)
    at = start + np.array([np.argmin(depth[a:a + k]) for a, k in zip(start, n)])
    return least, p_lat[at], p_lon[at]


def summary(bathy: Bathymetry, lat, lon, on_link=None, min_depth: float = 15.0) -> dict:
    """Legs that run over water min_depth m deep or less (links aside), and the depth at the two ends."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    on_link = np.zeros(len(lat) - 1, bool) if on_link is None else np.asarray(on_link, bool)
    least, w_lat, w_lon = shallowest(bathy, lat, lon)
    shallow = np.nonzero((least <= min_depth) & ~on_link)[0]
    ends = -bathy.elevation(lat[[0, -1]], lon[[0, -1]]).astype(float)
    return {
        "legs": int((~on_link).sum()),
        "shallow_legs": len(shallow),
        "least_m": round(float(least[~on_link].min()), 1) if (~on_link).any() else None,
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
    args = parser.parse_args()

    bathy = Bathymetry(args.gebco)
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        print(path)
        for row in rows:
            for key, found in row.items():
                if isinstance(found, dict) and "lat" in found:
                    s = summary(bathy, found["lat"], found["lon"], found.get("on_link"), args.min_depth)
                    print(f"  {row.get('route', '')} {key}: {s['shallow_legs']} of {s['legs']} legs ≤{args.min_depth:g} m "
                          f"(least {s['least_m']} m, start {s['start_m']:g} m, goal {s['goal_m']:g} m) {s['where'][:10]}")


if __name__ == "__main__":
    main()
