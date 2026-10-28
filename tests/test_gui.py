import json
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from shiprouting.fuel.ships import KCS_SHIP
from shiprouting.gui.server import Jobs, handler, planner
from test_fuel_route import T0, forecast
from test_plan import GOAL, GRAPH, LAND, START, coarse_tables  # noqa: F401 (fixture)


def wait(jobs, key):
    while jobs.get(key)["state"] == "running":
        time.sleep(0.01)
    return jobs.get(key)


def test_one_search_runs_at_a_time_and_its_printed_lines_are_kept():
    go = threading.Event()

    def run(request):
        print("price 1000.0 kg/h")
        go.wait()
        print("done", request["n"])
        return {"n": request["n"]}

    jobs = Jobs(run)
    key = jobs.start({"n": 1})
    assert jobs.start({"n": 2}) is None
    go.set()
    job = wait(jobs, key)
    assert job["result"] == {"n": 1} and job["log"] == ["price 1000.0 kg/h", "done 1"] and job["error"] is None
    assert jobs.start({"n": 3}) is not None


def test_a_failed_search_reports_its_error():
    jobs = Jobs(lambda request: 1 / 0)
    assert wait(jobs, jobs.start({}))["error"].startswith("ZeroDivisionError")


def test_the_planner_returns_the_routes_the_page_draws():
    fc = forecast(wind_u=-10.0, Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    asked = []
    runs = types.SimpleNamespace(newest=lambda t: (asked.append(t), (T0 - np.timedelta64(8, "h"), fc))[1])
    run = planner(GRAPH, LAND, runs, KCS_SHIP)
    request = {"start": START, "goal": GOAL, "departure": str(T0), "deadline": str(T0 + np.timedelta64(40, "h"))}
    shortest = run(request | {"kind": "shortest"})
    assert "least_fuel" not in shortest and set(shortest["shortest"]) >= {"km", "hours", "fuel_t", "lat", "lon"}
    assert asked == [T0] and shortest["run"] == str(T0 - np.timedelta64(8, "h"))  # the run out by the departure
    both = run(request | {"kind": "least_fuel"})
    assert both["least_fuel"]["fuel_t"] <= both["shortest"]["fuel_t"] and "tries" not in both


def test_the_page_and_a_search_are_served():
    jobs = Jobs(lambda request: {"echo": request["start"]})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(jobs, {"forecast_start": str(T0)}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"<title>" in urllib.request.urlopen(url + "/").read()
        assert json.load(urllib.request.urlopen(url + "/api/info")) == {"forecast_start": str(T0)}
        post = urllib.request.Request(url + "/api/plan", data=json.dumps({"start": [1, 2]}).encode(), method="POST")
        key = json.load(urllib.request.urlopen(post))["job"]
        wait(jobs, key)
        assert json.load(urllib.request.urlopen(f"{url}/api/jobs/{key}"))["result"] == {"echo": [1, 2]}
        try:
            urllib.request.urlopen(url + "/api/jobs/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
