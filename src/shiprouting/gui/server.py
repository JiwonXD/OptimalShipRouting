"""A map in the browser to plan one voyage: click the start, the goal and any waypoints, then
find the shortest route or the least-fuel one (route.plan) and see both drawn.

Serves on localhost. The sea graph and the forecast are loaded once at start. A search takes
minutes and most of the memory, so one runs at a time, in a thread the page polls for the
lines it prints and, at the end, the plan.
"""

import contextlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).with_name("static")
ROUTE_KEYS = ("km", "hours", "fuel_t", "speeds_kn", "links", "lat", "lon", "on_link", "segment_kn")


class Lines:
    """A file for print that hands each finished line to add."""

    def __init__(self, add):
        self.add, self.part = add, ""

    def write(self, text: str) -> int:
        *done, self.part = (self.part + text).split("\n")
        for line in done:
            if line.strip():
                self.add(line)
        return len(text)

    def flush(self) -> None:
        pass


class Jobs:
    """Searches run one at a time in a thread: run(request) returns the plan, and what it prints
    is kept as the job's log."""

    def __init__(self, run):
        self.run, self.jobs, self.lock = run, {}, threading.Lock()

    def start(self, request: dict) -> str | None:
        """The new job's id, or None while another is running."""
        with self.lock:
            if any(j["state"] == "running" for j in self.jobs.values()):
                return None
            key = uuid.uuid4().hex[:8]
            self.jobs[key] = {"state": "running", "log": [], "result": None, "error": None, "started": time.time()}
        threading.Thread(target=self._work, args=(self.jobs[key], request), daemon=True).start()
        return key

    def _work(self, job: dict, request: dict) -> None:
        try:
            with contextlib.redirect_stdout(Lines(job["log"].append)):  # one job at a time, so stdout is its alone
                job["result"] = self.run(request)
        except Exception as e:  # shown on the page
            job["error"] = f"{type(e).__name__}: {e}"
        job["seconds"] = round(time.time() - job["started"], 1)
        job["state"] = "done"

    def get(self, key: str) -> dict | None:
        return self.jobs.get(key)


def planner(graph, land, forecast, ship, tables: str | None = None):
    """run for Jobs: a request {start, goal, via, departure, deadline, kind: "shortest" | "least_fuel"}
    planned with route.plan, the routes cut down to what the page draws."""
    from shiprouting.route.plan import plan

    def run(request: dict) -> dict:
        out = plan(graph, ship, forecast, tuple(request["start"]), tuple(request["goal"]), request["departure"],
                   request["deadline"], tables=tables, via=[tuple(p) for p in request.get("via", [])], land=land,
                   least_fuel=request["kind"] == "least_fuel")
        for key in ("shortest", "least_fuel"):
            if key in out:
                out[key] = {k: out[key][k] for k in ROUTE_KEYS}
        out.pop("tries", None)
        return out

    return run


def handler(jobs: Jobs, info: dict):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            elif self.path == "/api/info":
                self.json(200, info)
            elif self.path.startswith("/api/jobs/"):
                job = jobs.get(self.path.rsplit("/", 1)[-1])
                self.json(200, job) if job else self.json(404, {"error": "no such job"})
            else:
                self.json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/plan":
                return self.json(404, {"error": "not found"})
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = jobs.start(request)
            self.json(202, {"job": key}) if key else self.json(409, {"error": "a search is already running"})

        def json(self, code: int, body) -> None:
            self.send(code, json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # no line per poll
            pass

    return Handler


def main() -> None:
    import argparse
    import glob

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.route.plan import add_sea_args, sea_graph
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Plan a voyage on a map in the browser.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("--tables", help="folder the wave added resistance tables are kept in")
    parser.add_argument("--port", type=int, default=8000)
    add_sea_args(parser)
    args = parser.parse_args()

    land, graph = sea_graph(args)
    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    info = {"forecast_start": str(forecast.time[0]), "forecast_end": str(forecast.time[-1])}
    jobs = Jobs(planner(graph, land, forecast, KCS_SHIP, args.tables))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(jobs, info))
    print(f"open http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
