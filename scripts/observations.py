"""Wind and wave observations against the weather the voyages are planned and judged in: the
ECMWF forecasts by lead time, and ERA5.

    python scripts/observations.py buoys OUT --stations 46006,41048,... --months 2025-12,2026-01,...
    python scripts/observations.py altimeters OUT --months 2025-12,2026-01,... [--missions s3a,j3,...]
    python scripts/observations.py compare [--buoys DIR] [--altimeters DIR] --runs DIR [--runs DIR] --era5 DIR --out table.csv

buoys: moored buoys of NOAA NDBC (public domain): the station table and the standard
meteorological files covering the months (the yearly file of a past year, the monthly files of
this one, and the last 45 days), one text file per station.
altimeters: significant wave height along the satellite tracks, Copernicus Marine
WAVE_GLO_PHY_SWH_L3_NRT_014_001 (1 Hz, filtered VAVH; needs copernicusmarine and a login),
averaged over blocks of BLOCK_S seconds (about 40 km of track, near the 0.25° model grid), one
file per mission and month.
compare: every hourly buoy observation of significant wave height (WVHT) and wind speed (WSPD),
and every altimeter block of Hs, against each source at the place, bilinear in space and
linear in time. The buoys measure wind at their anemometer, about 4 m up on the common 3 m
discus hulls, so it is taken to 10 m with a neutral log profile (z0 = 0.2 mm); NDBC does not
list the height in the files, so this is an approximation and the wind bias is reported as
such. Wave period is not compared: the buoys give the mean period from the second moment (APD,
T_m02), the models T_m-1,0.
Rows: observations, source (ecmwf by lead time band, era5), variable, n, bias, rmse, scatter index, r.
"""

import argparse
import csv
import gzip
import os
import urllib.request

import numpy as np

NDBC = "https://www.ndbc.noaa.gov/data"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ANEMOMETER_M, Z0_M = 4.1, 0.0002
BLOCK_S = 6
MISSIONS = "al,c2,cfo,h2b,h2c,j3,s3a,s3b,s6a,swon"
LEADS = [(0, 21), (24, 45), (48, 69), (72, 93), (96, 141), (144, 237)]  # forecast hours, inclusive


def get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            raw = r.read()
    except OSError:
        return None
    return gzip.decompress(raw) if url.endswith(".gz") else raw


def read_stdmet(text: str) -> dict[str, np.ndarray]:
    """time (datetime64[s]), WSPD (m/s), WVHT (m) of an NDBC standard meteorological file; NaN where missing."""
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not rows:
        return {"time": np.array([], "datetime64[s]"), "WSPD": np.array([]), "WVHT": np.array([])}
    header = next(line.lstrip("#").split() for line in text.splitlines() if line.startswith("#"))
    col = {name: k for k, name in enumerate(header)}
    a = np.array([r[: len(header)] for r in rows if len(r) >= len(header)])
    num = lambda name: np.array([np.nan if v == "MM" else float(v) for v in a[:, col[name]]])
    yy = num("YY").astype(int)
    time = np.array([f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}" for y, m, d, h, mi in
                     zip(yy, num("MM").astype(int), num("DD").astype(int), num("hh").astype(int), num("mm").astype(int))], "datetime64[s]")
    wspd, wvht = num("WSPD"), num("WVHT")
    wspd[wspd >= 99] = np.nan  # the yearly files mark missing values with 99, 999 ...
    wvht[wvht >= 99] = np.nan
    order = np.argsort(time, kind="stable")
    return {"time": time[order], "WSPD": wspd[order], "WVHT": wvht[order]}


def stations(table: str) -> dict[str, tuple[float, float]]:
    """Station id -> (lat, lon) from NDBC's station_table.txt."""
    out = {}
    for line in table.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        sid, loc = parts[0].strip().lower(), parts[6]
        try:
            la, ns, lo, ew = loc.split("(")[0].split()[:4]
            out[sid] = (float(la) * (1 if ns == "N" else -1), float(lo) * (1 if ew == "E" else -1))
        except ValueError:
            continue
    return out


def cmd_buoys(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    table = get(f"{NDBC}/stations/station_table.txt")
    with open(os.path.join(args.out, "station_table.txt"), "wb") as f:
        f.write(table)
    months = [np.datetime64(m, "M") for m in args.months.split(",")]
    this_year = int(str(np.datetime64("today", "Y")))
    for sid in args.stations.lower().split(","):
        parts = []
        for year in sorted({int(str(m)[:4]) for m in months}):
            if year < this_year:
                parts.append(get(f"{NDBC}/historical/stdmet/{sid}h{year}.txt.gz"))
        for m in months:
            year, month = int(str(m)[:4]), int(str(m)[5:7])
            if year == this_year:  # this year's months: one file each, a while after the month ends
                parts.append(get(f"{NDBC}/stdmet/{MONTHS[month - 1]}/{sid}{'123456789abc'[month - 1]}{year}.txt.gz"))
        parts.append(get(f"{NDBC}/realtime2/{sid.upper()}.txt"))
        obs = [read_stdmet(p.decode("ascii", "replace")) for p in parts if p]
        if not obs:
            print(f"{sid}: nothing found", flush=True)
            continue
        rows = {}
        for o in obs:  # later files (the last 45 days) win on the same time
            for t, w, h in zip(o["time"], o["WSPD"], o["WVHT"]):
                if np.datetime64(t, "M") in months:
                    rows[t] = (w, h)
        with open(os.path.join(args.out, f"{sid}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "WSPD", "WVHT"])
            for t in sorted(rows):
                w.writerow([str(t), *rows[t]])
        print(f"{sid}: {len(rows)} observations", flush=True)


def load_buoys(folder: str) -> dict[str, np.ndarray]:
    """lat, lon, time, wind at 10 m and Hs of every hourly observation in folder."""
    with open(os.path.join(folder, "station_table.txt"), encoding="ascii", errors="replace") as f:
        where = stations(f.read())
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv") or name[:-4] not in where:
            continue
        with open(os.path.join(folder, name)) as f:
            rows = list(csv.DictReader(f))
        t = np.array([r["time"] for r in rows], "datetime64[s]")
        top = t.astype("datetime64[m]").astype(int) % 60 == 0  # whole hours only, as the models
        la, lo = where[name[:-4]]
        wind = np.array([float(r["WSPD"]) for r in rows]) * np.log(10.0 / Z0_M) / np.log(ANEMOMETER_M / Z0_M)
        hs = np.array([float(r["WVHT"]) for r in rows])
        out += [(la, lo, tt, ww, hh) for tt, ww, hh, ok in zip(t, wind, hs, top) if ok]
    return {"lat": np.array([o[0] for o in out]), "lon": np.array([o[1] for o in out]),
            "time": np.array([o[2] for o in out], "datetime64[s]"), "wind": np.array([o[3] for o in out]),
            "Hs": np.array([o[4] for o in out])}


def blocks(time, lat, lon, hs, seconds: int = BLOCK_S) -> dict[str, np.ndarray]:
    """Means over blocks of seconds of one track (time datetime64[s], ascending), blocks with
    all their points; NaN Hs points left out."""
    ok = np.isfinite(hs)
    time, lat, lon, hs = time[ok], lat[ok], lon[ok], hs[ok]
    key = time.astype(np.int64) // seconds
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    count = np.diff(np.r_[starts, len(key)])
    full = count == seconds
    mean = lambda v: (np.add.reduceat(v, starts) / count)[full]
    x, y = np.cos(np.radians(lon)), np.sin(np.radians(lon))  # across the dateline
    return {"time": (key[starts][full] * seconds + seconds // 2).astype("datetime64[s]"), "lat": mean(lat),
            "lon": np.degrees(np.arctan2(mean(y), mean(x))), "Hs": mean(hs)}


def cmd_altimeters(args) -> None:
    import glob
    import tempfile

    import copernicusmarine
    import xarray as xr

    os.makedirs(args.out, exist_ok=True)
    for mission in args.missions.split(","):
        dataset = f"cmems_obs-wave_glo_phy-swh_nrt_{mission}-l3_PT1S"
        for month in args.months.split(","):
            path = os.path.join(args.out, f"{mission}_{month}.npz")
            if os.path.exists(path):
                continue
            with tempfile.TemporaryDirectory() as tmp:
                yyyymm = month.replace("-", "")
                copernicusmarine.get(dataset_id=dataset, filter=f"*_{mission}_{yyyymm}[0-3][0-9]T*", output_directory=tmp, no_directories=True)
                parts = []
                for nc in sorted(glob.glob(os.path.join(tmp, "*.nc"))):
                    with xr.open_dataset(nc, engine="h5netcdf") as ds:
                        parts.append(blocks(ds["time"].values.astype("datetime64[s]"), ds["latitude"].values.astype(float),
                                            ds["longitude"].values.astype(float), ds["VAVH"].values.astype(float)))
            if not parts:
                print(f"{mission} {month}: nothing found", flush=True)
                continue
            out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            order = np.argsort(out["time"], kind="stable")
            np.savez_compressed(path, **{k: v[order] for k, v in out.items()})
            print(f"{mission} {month}: {len(order)} blocks from {len(parts)} files", flush=True)


def load_altimeters(folder: str) -> dict[str, np.ndarray]:
    parts = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".npz"):
            with np.load(os.path.join(folder, name)) as f:
                parts.append({k: f[k] for k in ("time", "lat", "lon", "Hs")})
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("time", "lat", "lon", "Hs")}
    out["wind"] = np.full(len(out["time"]), np.nan)
    return out


def model(fc, lat, lon, t):
    """10 m wind speed and Hs of Forecast fc at the observations (NaN outside its times)."""
    inside = (t >= fc.time[0]) & (t <= fc.time[-1])
    wind, hs = np.full(len(t), np.nan), np.full(len(t), np.nan)
    if inside.any():
        a = (lat[inside], lon[inside], t[inside])
        wind[inside] = np.hypot(fc.sample("wind_u", *a), fc.sample("wind_v", *a))
        hs[inside] = fc.sample("Hs", *a)
    return wind, hs


def scores(m, o) -> dict:
    ok = np.isfinite(m) & np.isfinite(o)
    m, o = m[ok], o[ok]
    if len(o) < 2:
        return {"n": len(o)}
    return {"n": len(o), "obs_mean": round(float(o.mean()), 3), "bias": round(float((m - o).mean()), 3),
            "rmse": round(float(np.sqrt(((m - o) ** 2).mean())), 3), "si": round(float(np.sqrt(((m - o - (m - o).mean()) ** 2).mean()) / o.mean()), 3),
            "r": round(float(np.corrcoef(m, o)[0, 1]), 3)}


def cmd_compare(args) -> None:
    from shiprouting.route.voyage import Runs
    from shiprouting.weather import era5

    obs = {}
    if args.buoys:
        obs["buoys"] = load_buoys(args.buoys)
    if args.altimeters:
        obs["altimeters"] = load_altimeters(args.altimeters)
    for kind, o in obs.items():
        print(f"{kind}: {len(o['time'])} observations", flush=True)
    by = {}  # (observations, source, variable) -> list of (model, observed)

    def add(kind, source, o, sel, mw, mh):
        by.setdefault((kind, source, "wind"), []).append((mw, o["wind"][sel]))
        by.setdefault((kind, source, "Hs"), []).append((mh, o["Hs"][sel]))

    for root in args.runs:
        runs = Runs(root, lag_hours=0.0)
        for run in runs.times:
            _, fc = runs.newest(run)
            for kind, o in obs.items():
                lead = (o["time"] - run) / np.timedelta64(1, "h")
                for lo, hi in LEADS:
                    sel = (lead >= lo) & (lead <= hi)
                    add(kind, f"ecmwf {lo}-{hi} h", o, sel, *model(fc, o["lat"][sel], o["lon"][sel], o["time"][sel]))
            print(f"ecmwf {run}", flush=True)
    days = sorted({np.datetime64(x, "D") for o in obs.values() for x in o["time"]})
    for day in days:
        files = [os.path.join(args.era5, f"era5_{str(d).replace('-', '')}.npz") for d in (day, day + 1)]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            continue
        fc = era5.load(files)
        for kind, o in obs.items():
            sel = o["time"].astype("datetime64[D]") == day
            add(kind, "era5", o, sel, *model(fc, o["lat"][sel], o["lon"][sel], o["time"][sel]))
    rows = [{"observations": k, "source": s, "variable": v} | scores(np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))
            for (k, s, v), pairs in by.items()]
    rows = [r for r in rows if r["n"]]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["observations", "source", "variable", "n", "obs_mean", "bias", "rmse", "si", "r"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Buoy and altimeter observations against the forecasts and ERA5.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("buoys")
    p.add_argument("out")
    p.add_argument("--stations", required=True)
    p.add_argument("--months", required=True, help="YYYY-MM, comma-separated")
    p = sub.add_parser("altimeters")
    p.add_argument("out")
    p.add_argument("--months", required=True, help="YYYY-MM, comma-separated")
    p.add_argument("--missions", default=MISSIONS)
    p = sub.add_parser("compare")
    p.add_argument("--buoys", help="folder written by `buoys`")
    p.add_argument("--altimeters", help="folder written by `altimeters`")
    p.add_argument("--runs", action="append", required=True, help="forecast runs folder (repeatable)")
    p.add_argument("--era5", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"buoys": cmd_buoys, "altimeters": cmd_altimeters, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
