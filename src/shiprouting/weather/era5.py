"""ERA5 reanalysis (ECMWF / Copernicus Climate Change Service, CC BY 4.0) from Google's public
ARCO-ERA5 copy, which needs no account: download and read into a Forecast.

ARCO-ERA5 keeps ERA5 hourly on a 0.25° grid as a Zarr store, one chunk per variable and hour
covering the whole globe (lz4 Blosc, float32, NaN where a field has no value, e.g. waves on
land). Only those chunks are fetched, over plain HTTPS. Variables used, ERA5 short names:
  u10, v10 (10 m wind), swh (significant height of wind sea and swell combined),
  mwd (mean direction waves come from, degrees true), mwp (mean period from the m-1 moment,
  T_m-1,0; the same as in the ECMWF open data forecasts).
ERA5 has no ocean currents. Recent months are ERA5T, preliminary data replaced later.

Downloading needs the optional ``numcodecs`` package (``pip install -e '.[weather]'``).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from shiprouting.weather.forecast import Forecast

STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ORIGIN = np.datetime64("1900-01-01T00", "h")  # the store's time index is hours since this
VARIABLES = {  # ARCO-ERA5 name -> name in the files written here
    "10m_u_component_of_wind": "wind_u",
    "10m_v_component_of_wind": "wind_v",
    "significant_height_of_combined_wind_waves_and_swell": "Hs",
    "mean_wave_direction": "mwd",
    "mean_wave_period": "T_m10",
}
NLAT, NLON = 721, 1440  # latitude 90 to -90, longitude 0 to 359.75


def _chunk(name: str, hour: np.datetime64) -> np.ndarray:
    """One variable at one hour, (NLAT, NLON) float32, as stored (latitude descending, longitude from 0)."""
    import urllib.request

    import numcodecs

    k = int((hour - ORIGIN) / np.timedelta64(1, "h"))
    with urllib.request.urlopen(f"{STORE}/{name}/{k}.0.0", timeout=120) as r:
        raw = r.read()
    return np.frombuffer(numcodecs.Blosc().decode(raw), "<f4").reshape(NLAT, NLON)


def download(day: np.datetime64, target_dir, every_hours: int = 3, workers: int = 8) -> Path:
    """The VARIABLES at 00, every_hours, ... UTC of one day, into target_dir/era5_YYYYMMDD.npz:
    time, then each variable as (time, NLAT, NLON) float32 as stored. A file already there is
    kept, so an interrupted download resumes."""
    day = np.datetime64(day, "D")
    path = Path(target_dir) / f"era5_{str(day).replace('-', '')}.npz"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = day + np.arange(0, 24, every_hours).astype("timedelta64[h]")
    jobs = [(name, h) for name in VARIABLES for h in hours]
    with ThreadPoolExecutor(workers) as pool:
        got = dict(zip(jobs, pool.map(lambda job: _chunk(*job), jobs)))
    arrays = {short: np.stack([got[name, h] for h in hours]) for name, short in VARIABLES.items()}
    part = path.with_suffix(".part.npz")
    np.savez_compressed(part, time=hours.astype("datetime64[s]"), **arrays)
    os.replace(part, path)
    return path


def load(paths, dtype=np.float32) -> Forecast:
    """Read files written by download into one Forecast (latitude ascending, longitude from -180),
    its fields of dtype. Wave direction becomes the east and north components of the unit vector
    pointing where the waves come from (wave_from_x, wave_from_y), as in ecmwf.load."""
    lat = np.linspace(90.0, -90.0, NLAT)[::-1]
    lon = (np.arange(NLON) * 0.25 + 180.0) % 360.0 - 180.0
    order_lon = np.argsort(lon)
    paths = sorted(paths)
    times = []
    for path in paths:
        with np.load(path) as f:
            times.append(f["time"])
    time = np.concatenate(times)
    if np.any(np.diff(time) <= np.timedelta64(0, "s")):
        raise ValueError("the files' times overlap or are out of order")
    fields = {short: np.empty((len(time), NLAT, NLON), dtype) for short in VARIABLES.values()}  # filled in place: one copy
    k = 0
    for path, t in zip(paths, times):
        with np.load(path) as f:
            for short in VARIABLES.values():
                fields[short][k : k + len(t)] = f[short][:, ::-1][:, :, order_lon]
        k += len(t)
    rad = np.radians(fields.pop("mwd"))
    fields["wave_from_x"], fields["wave_from_y"] = np.sin(rad).astype(dtype), np.cos(rad).astype(dtype)
    return Forecast(lat=lat, lon=lon[order_lon], time=time, fields=fields)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download ERA5 days from ARCO-ERA5, one file a day.")
    parser.add_argument("first", help="first day, YYYY-MM-DD")
    parser.add_argument("last", help="last day, YYYY-MM-DD")
    parser.add_argument("target", help="folder for the files")
    parser.add_argument("--every-hours", type=int, default=3)
    args = parser.parse_args()
    day, last = np.datetime64(args.first, "D"), np.datetime64(args.last, "D")
    while day <= last:
        print(download(day, args.target, args.every_hours), flush=True)
        day += np.timedelta64(1, "D")


if __name__ == "__main__":
    main()
