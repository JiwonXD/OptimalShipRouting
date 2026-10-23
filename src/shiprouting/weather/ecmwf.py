"""ECMWF open data (IFS, CC BY 4.0): download and read into a Forecast.

Variables used, with ECMWF short names:
  oper stream: 10u, 10v (10 m wind), sve, svn (surface current)
  wave stream: swh (significant wave height of wind sea and swell combined),
               mwd (mean direction waves come from, degrees true),
               mwp (mean period from the m-1 moment, T_m-1,0)

Reading needs the optional ``eccodes`` package and downloading needs
``ecmwf-opendata`` (``pip install -e '.[weather]'``).
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from shiprouting.weather.forecast import Forecast

PARAMS = {"oper": ("10u", "10v", "sve", "svn"), "wave": ("swh", "mwd", "mwp")}

# ECMWF short name -> Forecast field name.
NAMES = {
    "10u": "wind_u",
    "10v": "wind_v",
    "sve": "current_u",
    "svn": "current_v",
    "swh": "Hs",
    "mwp": "T_m10",
}


def download(run: datetime, steps, target_dir, resol: str = "0p25", source: str = "ecmwf") -> list[Path]:
    """Download the PARAMS of forecast ``run`` (UTC) at ``steps`` (hours), one file per stream and step.

    source "ecmwf" keeps only the last few days of runs; "aws" (the AWS open data mirror)
    keeps older ones.
    """
    from ecmwf.opendata import Client

    client = Client(source=source, resol=resol)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stream, params in PARAMS.items():
        for step in steps:
            path = target_dir / f"ifs{resol}_{run:%Y%m%dT%H}z_{stream}_step{step:03d}.grib2"
            client.retrieve(
                date=run.strftime("%Y%m%d"),
                time=run.hour,
                step=step,
                stream=stream,
                type="fc",
                param=list(params),
                target=str(path),
            )
            paths.append(path)
    return paths


def load(paths) -> Forecast:
    """Read GRIB2 files with the PARAMS into one Forecast.

    Every variable must be present at every valid time, on the same regular
    global grid. Wave direction is stored as the east and north components of
    the unit vector pointing where the waves come from (wave_from_x,
    wave_from_y), so it interpolates without a jump at 360°.
    """
    import eccodes

    grids = {}  # (field, valid time) -> 2-D array, latitude ascending
    lat = lon = None
    for path in paths:
        with open(path, "rb") as f:
            while (h := eccodes.codes_grib_new_from_file(f)) is not None:
                try:
                    short = eccodes.codes_get(h, "shortName")
                    if short not in NAMES and short != "mwd":
                        continue
                    valid = np.datetime64(
                        datetime.strptime(
                            f"{eccodes.codes_get(h, 'validityDate')}{eccodes.codes_get(h, 'validityTime'):04d}",
                            "%Y%m%d%H%M",
                        ),
                        "s",
                    )
                    m_lat, m_lon, values = _read_grid(eccodes, h)
                finally:
                    eccodes.codes_release(h)
                if lat is None:
                    lat, lon = m_lat, m_lon
                elif not (np.array_equal(lat, m_lat) and np.array_equal(lon, m_lon)):
                    raise ValueError(f"{path}: {short} is on a different grid")
                if short == "mwd":
                    rad = np.radians(values)
                    grids["wave_from_x", valid] = np.sin(rad)
                    grids["wave_from_y", valid] = np.cos(rad)
                else:
                    grids[NAMES[short], valid] = values

    names = sorted({name for name, _ in grids})
    times = np.array(sorted({t for _, t in grids}))
    missing = [(n, t) for n in names for t in times if (n, t) not in grids]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    fields = {n: np.stack([grids[n, t] for t in times]) for n in names}
    return Forecast(lat=lat, lon=lon, time=times, fields=fields)


def _read_grid(eccodes, h):
    """Latitudes (ascending), longitudes (from -180, ascending) and values (NaN where missing)."""
    if eccodes.codes_get(h, "gridType") != "regular_ll":
        raise ValueError("only regular_ll grids are supported")
    ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
    lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
    lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
    dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
    dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
    if not math.isclose(ni * dlon, 360.0):
        raise ValueError("grid does not cover all longitudes")
    if eccodes.codes_get(h, "jScansPositively"):
        lat = lat0 + dlat * np.arange(nj)
    else:
        lat = lat0 - dlat * np.arange(nj)
    lon = (lon0 + dlon * np.arange(ni) + 180.0) % 360.0 - 180.0

    values = eccodes.codes_get_values(h).astype(np.float64).reshape(nj, ni)
    if eccodes.codes_get(h, "bitmapPresent"):
        values[values == eccodes.codes_get(h, "missingValue")] = np.nan

    order_lat = np.argsort(lat)
    order_lon = np.argsort(lon)
    return lat[order_lat], lon[order_lon], values[np.ix_(order_lat, order_lon)]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download an ECMWF open data run, or the first hours of daily runs.")
    parser.add_argument("run", help="the (first) run, YYYYMMDDHH UTC")
    parser.add_argument("target", help="folder for the GRIB2 files")
    parser.add_argument("--days", type=int, default=0, help="instead of the whole run, steps 0-21 h of this many daily runs from `run` on")
    parser.add_argument("--source", default="ecmwf", help="ecmwf (last few days) or aws (older runs)")
    args = parser.parse_args()

    run = datetime.strptime(args.run, "%Y%m%d%H")
    if args.days:
        for day in range(args.days):
            download(run + timedelta(days=day), range(0, 24, 3), args.target, source=args.source)
    else:
        download(run, [*range(0, 144, 3), *range(144, 361, 6)], args.target, source=args.source)


if __name__ == "__main__":
    main()
