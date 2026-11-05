"""Edge cases for the map GUI: send each scenario to a running gui.server, keep what comes back,
then (with --check) look at the routes for land, shallow water and other faults.

    python scripts/gui_scenarios.py run out.jsonl [--only name,name] [--url http://127.0.0.1:8000]
    python scripts/gui_scenarios.py check out.jsonl LAND_SHP --bathymetry GEBCO.nc
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np

PORTS = {
    "busan": (35.08, 129.08), "la": (33.72, -118.27), "rotterdam": (51.95, 4.05), "singapore": (1.26, 103.84),
    "newyork": (40.6, -74.0), "chennai": (13.1, 80.3), "odessa": (46.5, 30.7), "piraeus": (37.9, 23.6),
    "dubai": (25.27, 55.29), "stpetersburg": (59.9, 30.2), "shanghai": (31.23, 121.8), "auckland": (-36.8, 174.8),
    "honolulu": (21.3, -157.87), "vancouver": (49.29, -123.11), "santos": (-23.96, -46.3),
}
D0, D1 = "2026-08-24T00:00", "2026-09-23T00:00"  # the GUI's defaults


def s(name, start, goal, via=(), departure=D0, deadline=D1, kind="least_fuel", note=""):
    p = lambda v: list(PORTS[v]) if isinstance(v, str) else list(v)  # noqa: E731
    return {"name": name, "note": note, "request": {"start": p(start), "goal": p(goal), "via": [p(v) for v in via],
                                                    "departure": departure, "deadline": deadline, "kind": kind}}


SCENARIOS = [
    # the clicked points
    s("land_seoul", (37.57, 126.98), "la", note="start inland, 30 km from the sea"),
    s("land_mongolia", (47.9, 106.9), "busan", kind="shortest", note="start 1,000+ km inland"),
    s("caspian", (42.0, 50.5), "busan", kind="shortest", note="sea with no way out"),
    s("great_lakes", (41.9, -87.6), "rotterdam", kind="shortest", note="Chicago: lakes are land in OSM"),
    s("shallow_start", (37.45, 126.3), "shanghai", kind="shortest", note="Incheon tidal flats, under 15 m"),
    s("bad_coords", (95.0, 400.0), "busan", kind="shortest", note="latitude 95, longitude 400"),
    s("antarctic", (-75.0, 0.0), "santos", kind="shortest", note="south of the ice-free limit (-60)"),
    s("same_point", "busan", "busan", note="start = goal"),
    s("near_points", "busan", (35.05, 129.13), note="goal 5 km from the start"),
    # time
    s("deadline_before", "busan", "la", deadline="2026-08-20T00:00", note="deadline before the departure"),
    s("deadline_equal", "busan", "la", deadline=D0, note="deadline = departure"),
    s("deadline_tight", "busan", "rotterdam", deadline="2026-08-29T00:00", note="5 days to Rotterdam"),
    s("deadline_just", "busan", "la", deadline="2026-09-04T00:00", note="LA in 11 days: ~20 kn needed"),
    s("deadline_loose", "busan", "la", deadline="2026-10-24T00:00", note="60 days: slowest speed, past the 15-day forecast"),
    s("depart_early", "busan", "la", departure="2026-08-01T00:00", deadline="2026-08-31T00:00", note="before the first run"),
    s("depart_late", "busan", "la", departure="2026-12-01T00:00", deadline="2026-12-31T00:00", note="after the last run"),
    s("depart_edge", "busan", "la", departure="2026-11-05T00:00", deadline="2026-11-25T00:00", note="last run's last days"),
    s("depart_odd_hour", "busan", "la", departure="2026-09-01T13:37", deadline="2026-09-20T00:00", note="between runs"),
    # the way
    s("dateline_south", "auckland", "honolulu", note="crosses 180°"),
    s("dateline_north", "busan", "vancouver", note="great circle near the Aleutians"),
    s("suez", "rotterdam", "singapore", note="Suez Canal"),
    s("panama", "newyork", "la", note="Panama Canal"),
    s("malacca", "busan", "chennai", note="Malacca or Sunda"),
    s("bosporus", "odessa", "piraeus", note="Bosporus + Dardanelles"),
    s("hormuz", "dubai", "singapore", note="Strait of Hormuz"),
    s("baltic", "stpetersburg", "rotterdam", note="Danish straits"),
    s("short_hop", "busan", "shanghai", deadline="2026-08-26T00:00", note="short coastal leg"),
    # waypoints
    s("via_many", "busan", "la", via=[(38, 145), (42, 155), (45, 165), (47, 175), (48, -175), (47, -165), (45, -155),
                                    (42, -145), (39, -135), (36, -125)], note="10 waypoints"),
    s("via_zigzag", "busan", "la", via=[(30, 160), (45, 170), (25, -170), (45, -150)], note="zigzag"),
    s("via_back", "busan", "la", via=[(30, -140), (35, 150)], note="waypoint past the goal, then back"),
    s("via_on_land", "busan", "la", via=[(36.2, 138.2)], note="waypoint in the middle of Honshu"),
    s("via_same_as_start", "busan", "la", via=["busan"], note="waypoint = start"),
    s("via_duplicate", "busan", "la", via=[(40, 160), (40, 160)], note="same waypoint twice"),
    s("via_polar", "rotterdam", "busan", via=[(78, 60), (76, 140)], kind="shortest", note="Northern Sea Route, past 72°N"),
    s("cape_horn", "newyork", "la", via=[(-57, -67)], deadline="2026-10-23T00:00", note="round Cape Horn (-57°)"),
]


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def run(args):
    todo = [x for x in SCENARIOS if not args.only or x["name"] in args.only.split(",")]
    done = set()
    try:
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["name"] for line in f if line.strip()}
    except FileNotFoundError:
        pass
    print("server:", call(args.url, "/api/info"), flush=True)
    for x in todo:
        if x["name"] in done:
            continue
        t0 = time.time()
        while True:
            code, body = call(args.url, "/api/plan", x["request"])
            if code != 409:
                break
            time.sleep(5)
        job = {"state": "not started", "post": [code, body]}
        if code == 202:
            while job["state"] != "done":
                time.sleep(2)
                job = call(args.url, f"/api/jobs/{body['job']}")[1]
        rec = {**x, "wall_s": round(time.time() - t0, 1), "job": job}
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        res = job.get("result") or {}
        print(f"{x['name']:18} {rec['wall_s']:7.1f}s error={job.get('error') or res.get('error')} moved={res.get('moved_km')} "
              + " ".join(f"{k}={res[k]['km']}km/{res[k]['hours']}h/{res[k]['fuel_t']}t" for k in ("shortest", "least_fuel") if k in res),
              flush=True)


def check(args):
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.sea.latlon import haversine_km, turn_deg
    from shiprouting.sea.quadtree import arcs_on_land

    land = load_osm_land_polygons(args.land)
    lands = {"land": land}
    if args.bathymetry:
        from shiprouting.sea.depth import Bathymetry, LandOrShallow, link_ends

        lands["land_or_shallow"] = LandOrShallow(land, Bathymetry(args.bathymetry), 15.0, link_ends(), 10.0)
    with open(args.out, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    for rec in recs:
        res, req = rec["job"].get("result") or {}, rec["request"]
        hours = (np.datetime64(req["deadline"]) - np.datetime64(req["departure"])) / np.timedelta64(1, "h")
        print(f"== {rec['name']} ({rec['note']}) wall {rec['wall_s']} s")
        print("   error:", rec["job"].get("error") or res.get("error") or "-", "| moved_km:", res.get("moved_km"), "| run:", res.get("run"),
              "| saving:", res.get("saving_t"), res.get("saving_pct"))
        for key in ("shortest", "least_fuel"):
            if key not in res:
                continue
            r = res[key]
            lat, lon, on_link = np.array(r["lat"]), np.array(r["lon"]), np.array(r["on_link"])
            seg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            free = ~on_link  # one per segment; a link's own segments are not checked
            hits = {name: np.nonzero(arcs_on_land(l, lat[:-1][free], lon[:-1][free], lat[1:][free], lon[1:][free]))[0]
                    for name, l in lands.items()}
            turns = turn_deg(lat, lon) if len(lat) > 2 else np.array([])
            kn = np.array(r["segment_kn"])
            print(f"   {key}: {r['km']} km {r['hours']} h (limit {hours:.1f}) {r['fuel_t']} t speeds {r['speeds_kn']} links {r['links']}"
                  f" | points {len(lat)} longest seg {seg.max() if len(seg) else 0:.0f} km | max |lat| {np.abs(lat).max():.1f}"
                  f" | turns>=60: {int((turns >= 60).sum())} | kn {kn.min() if len(kn) else '-'}..{kn.max() if len(kn) else '-'}"
                  f" | late {r['hours'] > hours + 0.05}")
            for name, k in hits.items():
                if len(k):
                    idx = np.nonzero(free)[0][k]
                    print(f"   !! {key} segments on {name}: {len(k)}, first at {[(round(lat[i], 3), round(lon[i], 3)) for i in idx[:5]]}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("out")
    a.add_argument("--only")
    a.add_argument("--url", default="http://127.0.0.1:8000")
    b = sub.add_parser("check")
    b.add_argument("out")
    b.add_argument("land")
    b.add_argument("--bathymetry")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(args) if args.cmd == "run" else check(args)


if __name__ == "__main__":
    main()
