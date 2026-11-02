"""Wind and wave observations from moored buoys (NOAA NDBC, public domain) against the weather the
voyages are planned and judged in: the ECMWF forecasts by lead time, and ERA5.

    python scripts/buoys.py fetch OUT --stations 46006,41048,... --months 2025-12,2026-01,2026-08,...
    python scripts/buoys.py compare OBS --runs DIR [--runs DIR] --era5 DIR --out table.csv

fetch: NDBC's station table and the standard meteorological files covering the months (the
yearly file of a past year, the monthly files of this one, and the last 45 days), one text
file per station with the hours of those months.
compare: every hourly observation of significant wave height (WVHT) and wind speed (WSPD)
against each source at the buoy, bilinear in space and linear in time. The buoys measure wind
at their anemometer, about 4 m up on the common 3 m discus hulls, so it is taken to 10 m with
a neutral log profile (z0 = 0.2 mm); NDBC does not list the height in the files, so this is an
approximation and the wind bias is reported as such. Wave period is not compared: the buoys
give the mean period from the second moment (APD, T_m02), the models T_m-1,0.
Rows: source (ecmwf by lead time band, era5), variable, n, bias, rmse, scatter index, r.
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


def cmd_fetch(args) -> None:
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


def load_obs(folder: str):
    """(station, lat, lon, time, wind at 10 m, Hs) of every hourly observation in folder."""
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
        out += [(name[:-4], la, lo, tt, ww, hh) for tt, ww, hh, ok in zip(t, wind, hs, top) if ok]
    st = np.array([o[0] for o in out])
    return st, np.array([o[1] for o in out]), np.array([o[2] for o in out]), np.array([o[3] for o in out], "datetime64[s]"), \
        np.array([o[4] for o in out]), np.array([o[5] for o in out])


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

    st, lat, lon, t, wind, hs = load_obs(args.obs)
    print(f"{len(t)} observations at {len(set(st))} buoys", flush=True)
    by = {}  # (source, variable) -> list of (model, obs)
    for root in args.runs:
        runs = Runs(root, lag_hours=0.0)
        for run in runs.times:
            _, fc = runs.newest(run)
            mw, mh = model(fc, lat, lon, t)
            lead = (t - run) / np.timedelta64(1, "h")
            for lo, hi in LEADS:
                sel = (lead >= lo) & (lead <= hi)
                by.setdefault((f"ecmwf {lo}-{hi} h", "wind"), []).append((mw[sel], wind[sel]))
                by.setdefault((f"ecmwf {lo}-{hi} h", "Hs"), []).append((mh[sel], hs[sel]))
            print(f"ecmwf {run}", flush=True)
    days = sorted({np.datetime64(x, "D") for x in t})
    for day in days:
        files = [os.path.join(args.era5, f"era5_{str(d).replace('-', '')}.npz") for d in (day, day + 1)]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            continue
        fc = era5.load(files)
        sel = t.astype("datetime64[D]") == day
        mw, mh = model(fc, lat[sel], lon[sel], t[sel])
        by.setdefault(("era5", "wind"), []).append((mw, wind[sel]))
        by.setdefault(("era5", "Hs"), []).append((mh, hs[sel]))
    rows = [{"source": s, "variable": v} | scores(np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))
            for (s, v), pairs in by.items()]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["source", "variable", "n", "obs_mean", "bias", "rmse", "si", "r"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Buoy observations against the forecasts and ERA5.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("out")
    p.add_argument("--stations", required=True)
    p.add_argument("--months", required=True, help="YYYY-MM, comma-separated")
    p = sub.add_parser("compare")
    p.add_argument("obs")
    p.add_argument("--runs", action="append", required=True, help="forecast runs folder (repeatable)")
    p.add_argument("--era5", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    (cmd_fetch if args.cmd == "fetch" else cmd_compare)(args)


if __name__ == "__main__":
    main()
