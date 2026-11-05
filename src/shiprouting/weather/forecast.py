"""Gridded forecast fields and their interpolation in space and time.

Fields live on a regular global latitude/longitude grid. Interpolation is
bilinear in space and linear in time; grid points without data (NaN, e.g. land
or sea ice in wave fields) are left out and the remaining weights rescaled, so
points next to a coast still get a value. A point with no valid neighbour is
NaN.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Forecast:
    lat: np.ndarray  # degrees, ascending, regular
    lon: np.ndarray  # degrees, ascending, regular, covering 360°
    time: np.ndarray  # datetime64[s], ascending, at least two
    fields: dict[str, np.ndarray]  # name -> (time, lat, lon) array

    def sample(self, name: str, lat, lon, time) -> np.ndarray:
        """Field ``name`` at ``lat``, ``lon`` (degrees) and ``time`` (datetime64), broadcast."""
        data = self.fields[name]
        t_grid = self.time.astype("datetime64[s]").astype(np.float64)
        lat, lon, t = np.broadcast_arrays(
            np.asarray(lat, np.float64),
            np.asarray(lon, np.float64),
            np.asarray(time, "datetime64[s]").astype(np.float64),
        )
        if np.any(t < t_grid[0]) or np.any(t > t_grid[-1]):
            raise ValueError("time is outside the forecast")

        k = np.clip(np.searchsorted(t_grid, t, side="right") - 1, 0, len(t_grid) - 2)
        wt = (t - t_grid[k]) / (t_grid[k + 1] - t_grid[k])

        dlat = self.lat[1] - self.lat[0]
        fy = (lat - self.lat[0]) / dlat
        j = np.clip(np.floor(fy).astype(int), 0, len(self.lat) - 2)
        wy = np.clip(fy - j, 0.0, 1.0)

        dlon = self.lon[1] - self.lon[0]
        fx = (lon - self.lon[0]) / dlon
        i0 = np.floor(fx).astype(int)
        wx = fx - i0
        i0 %= len(self.lon)
        i1 = (i0 + 1) % len(self.lon)  # wraps across the dateline

        num = np.zeros(lat.shape)
        den = np.zeros(lat.shape)
        for kk, wk in ((k, 1 - wt), (k + 1, wt)):
            for jj, wj in ((j, 1 - wy), (j + 1, wy)):
                for ii, wi in ((i0, 1 - wx), (i1, wx)):
                    v = data[kk, jj, ii]
                    w = wk * wj * wi
                    valid = ~np.isnan(v)
                    num += np.where(valid, w * v, 0.0)
                    den += np.where(valid, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)
