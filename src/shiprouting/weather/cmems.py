"""Surface currents from Copernicus Marine (global analysis and forecast, 1/12°, hourly means;
free with an account): download onto the ERA5 grid, to read with era5.load(..., currents=...).

Dataset cmems_mod_glo_phy_anfc_0.083deg_PT1H-m (product GLOBAL_ANALYSISFORECAST_PHY_001_024),
variables uo, vo (eastward and northward current at 0.49 m, m/s); past days are analyses. Each
0.25° point of the ERA5 grid takes the mean of the 3 x 3 model points around it (NaN where all
are land). Needs the optional ``copernicusmarine`` package and a login (``copernicusmarine login``).
"""

import os
from pathlib import Path

import numpy as np

from shiprouting.weather import era5

DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"


def to_era5_grid(lat, lon, u) -> np.ndarray:
    """Fields u (..., lat, lon) on a regular 1/12° grid (latitude and longitude ascending, longitude
    covering 360°) as the means of 3 x 3 points around each point of the ERA5 grid, laid out as
    ARCO-ERA5 stores it (latitude from 90 down, longitude from 0); NaN where none has a value
    or the model grid does not reach."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    out_lat = np.linspace(90.0, -90.0, era5.NLAT)
    out_lon = np.arange(era5.NLON) * 0.25
    j = np.rint((out_lat - lat[0]) * 12).astype(int)  # model row of each ERA5 latitude
    i = np.rint(((out_lon - lon[0]) % 360.0) * 12).astype(int) % len(lon)
    if not (np.allclose(np.diff(lat), 1 / 12, atol=1e-4) and np.allclose(np.diff(lon), 1 / 12, atol=1e-4)):
        raise ValueError("not a regular 1/12° grid")
    shape = (*u.shape[:-2], era5.NLAT, era5.NLON)
    total, count = np.zeros(shape), np.zeros(shape)
    for dj in (-1, 0, 1):
        jj = j + dj
        inside = (jj >= 0) & (jj < len(lat))
        for di in (-1, 0, 1):
            v = np.full(shape, np.nan)
            v[..., inside, :] = u[..., jj[inside], :][..., (i + di) % len(lon)]
            ok = ~np.isnan(v)
            total += np.where(ok, v, 0.0)
            count += ok
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan).astype("f4")


def download(day: np.datetime64, target_dir, every_hours: int = 3) -> Path:
    """Currents at 00, every_hours, ... UTC of one day into target_dir/cmems_YYYYMMDD.npz: time,
    current_u, current_v as (time, NLAT, NLON) float32 in ARCO-ERA5's layout. A file already
    there is kept."""
    import copernicusmarine

    day = np.datetime64(day, "D")
    path = Path(target_dir) / f"cmems_{str(day).replace('-', '')}.npz"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = day + np.arange(0, 24, every_hours).astype("timedelta64[h]")
    ds = copernicusmarine.open_dataset(dataset_id=DATASET, variables=["uo", "vo"], minimum_depth=0.0, maximum_depth=1.0,
                                       start_datetime=str(hours[0]), end_datetime=str(hours[-1]))
    ds = ds.isel(depth=0).sel(time=hours.astype("datetime64[ns]")).load()
    lat, lon = ds["latitude"].values, ds["longitude"].values
    arrays = {name: to_era5_grid(lat, lon, ds[var].values) for name, var in (("current_u", "uo"), ("current_v", "vo"))}
    part = path.with_suffix(".part.npz")
    np.savez_compressed(part, time=hours.astype("datetime64[s]"), **arrays)
    os.replace(part, path)
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Copernicus Marine surface currents onto the ERA5 grid, one file a day.")
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
