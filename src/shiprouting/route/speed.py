"""Route and speed together: the least fuel arriving by a given time, each move at one of a few speeds.

Every speed has its own FuelCost in moving weather. When the search expands a cell, each
move is costed at every speed, at the time the ship reaches the cell; the clock (hours)
advances by that move's time at the speed taken. Links are taken at calm-water rates.

Time is not part of the search state. Instead each hour is given a price in kg of fuel
(per_hour), each move takes the speed with the least fuel + price × hours, and the
search minimises that sum. Price 0 sails at the slowest speed; a higher price sails
faster. meet_eta bisects the price until the route arrives just within the given time.
"""

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np

from shiprouting.route import kernel
from shiprouting.route.fuel import KNOT, FuelCost, fuel_kg, stacked_tables
from shiprouting.sea.latlon import SearchResult, haversine_km
from shiprouting.sea.land import Land
from shiprouting.sea.quadtree import Graph, OpenWater, along, arcs_on_land, astar, join_legs, trace


@dataclass
class SpeedChoice:
    costs: list[FuelCost]  # one per speed, slowest first, moving weather, same departure
    per_hour: float = 0.0  # the price of an hour, kg of fuel
    _args: tuple | None = field(default=None, repr=False)  # for kernel.fuel_kg, all speeds

    def kg_hours(self, lat_a, lon_a, lat_b, lon_b, hours=0.0) -> tuple[np.ndarray, np.ndarray]:
        """FuelCost.kg_hours at every speed in one pass, each shape (speeds, *moves)."""
        if self._args is None:
            self._args = stacked_tables(self.costs)
        return fuel_kg(self.costs, self._args, lat_a, lon_a, lat_b, lon_b, hours)

    def kg_hours_at(self, lat_a, lon_a, lat_b, lon_b, hours, at) -> tuple[np.ndarray, np.ndarray]:
        """kg_hours of each move at its own speed index `at` only, 1-D: kg_hours(...)[:, at[m], m]."""
        if self._args is None:
            self._args = stacked_tables(self.costs)
        t0 = time.perf_counter()
        moves = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (lat_a, lon_a, lat_b, lon_b, hours)), np.asarray(at, np.int64))
        moves = [np.ascontiguousarray(v).ravel() for v in moves]
        out, out_h, counts = np.empty(moves[0].size), np.empty(moves[0].size), np.zeros((len(self.costs), 3), np.int64)
        kernel.fuel_kg_at(*moves, out, out_h, counts, *self._args)
        seconds = (time.perf_counter() - t0) / len(self.costs)
        for c, m in zip(self.costs, counts):
            c.no_waves += int(m[0])
            c.over_mcr += int(m[1])
            c.beyond_forecast += int(m[2])
            c.seconds += seconds
        return out, out_h

    def kg(self, lat_a, lon_a, lat_b, lon_b, hours=0.0) -> np.ndarray:
        """FuelCost.kg at every speed in one pass, shape (speeds, *moves)."""
        return self.kg_hours(lat_a, lon_a, lat_b, lon_b, hours)[0]

    def edge_cost(self, graph: Graph):
        """edge_cost function for quadtree.astar, its clock in hours: fuel + per_hour × hours."""

        def costs(c, hours):
            lo, hi = graph.indptr[c], graph.indptr[c + 1]
            n = graph.indices[lo:hi]
            kg, h = self.kg_hours(graph.lat[c], graph.lon[c], graph.lat[n], graph.lon[n], hours)  # (speeds, moves)
            total = kg + self.per_hour * h
            pick = np.argmin(total, axis=0)
            moves = np.arange(kg.shape[1])
            return total[pick, moves], h[pick, moves]

        return costs

    def search(self, graph: Graph, s: int, g: int, max_seconds: float = math.inf, clock0: float = 0.0,
               chunk: int = 100_000, heap: int | None = None) -> SearchResult:
        """astar(graph, s, g, self.edge_cost(graph), self.link_cost, lambda n: 0.0, max_seconds, clock0),
        compiled (kernel.search): the same path, clock and cost. The time limit is checked every
        `chunk` cells expanded; the heap starts with room for `heap` entries (twice the nodes)
        and doubles when full."""
        t0 = time.perf_counter()
        if self._args is None:
            self._args = stacked_tables(self.costs)
        speed = np.array([c.speed for c in self.costs])
        calm = np.array([c.calm_kg_per_km for c in self.costs])
        n_nodes = len(graph.lat)
        jumps = [[] for _ in range(n_nodes)] if graph.links else None
        for i, (a, b, link) in enumerate(graph.links):  # as astar builds its jumps
            km = (
                float(haversine_km(graph.lat[a], graph.lon[a], link.lat[0], link.lon[0]))
                + link.km
                + float(haversine_km(graph.lat[b], graph.lon[b], link.lat[-1], link.lon[-1]))
            )
            jumps[a].append((b, km, i, False))
            jumps[b].append((a, km, i, True))
        jptr = np.zeros(n_nodes + 1, np.int64)
        flat = []
        if jumps is not None:
            jptr[1:] = np.cumsum([len(j) for j in jumps])
            flat = [j for js in jumps for j in js]
        jto = np.array([j[0] for j in flat], np.int64)
        jkg = np.array([[j[1] * c for c in calm] for j in flat], float).reshape(len(flat), len(speed))
        jh = np.array([[j[1] / (v * 3.6) for v in speed] for j in flat], float).reshape(len(flat), len(speed))
        best, clock = np.full(n_nodes, np.inf), np.zeros(n_nodes)
        parent, via, closed = np.full(n_nodes, -1, np.int64), np.full(n_nodes, -1, np.int64), np.zeros(n_nodes, bool)
        hc, hn = np.empty(heap or 2 * n_nodes), np.empty(heap or 2 * n_nodes, np.int64)
        best[s], clock[s] = 0.0, clock0
        state, counts = np.array([1, 0], np.int64), np.zeros((len(speed), 3), np.int64)
        hc[0], hn[0] = 0.0, s
        indptr, indices = graph.indptr.astype(np.int64), graph.indices.astype(np.int64)
        while True:
            status = kernel.search(indptr, indices, graph.lat, graph.lon, jptr, jto, jkg, jh, g, float(self.per_hour), chunk,
                                   best, clock, parent, via, closed, hc, hn, state, counts, *self._args)  # fmt: skip
            if status == 1:
                break
            if status == 3:
                raise ValueError(f"no route ({state[1]} cells expanded)")
            if status == 2:
                hc, hn = np.r_[hc, np.empty(len(hc))], np.r_[hn, np.empty(len(hn), np.int64)]
            if time.perf_counter() - t0 > max_seconds:
                raise TimeoutError(f"stopped after {max_seconds:g} s ({state[1]} cells expanded)")
        for c, m in zip(self.costs, counts):
            c.no_waves += int(m[0])
            c.over_mcr += int(m[1])
            c.beyond_forecast += int(m[2])
        back, n = {s: None}, g
        while n != s:
            j = via[n]
            back[n] = (int(parent[n]), None, False) if j < 0 else (int(parent[n]), graph.links[flat[j][2]][2], flat[j][3])
            n = int(parent[n])
        return trace(graph, s, g, back, clock, float(best[g]), int(state[1]), time.perf_counter() - t0)

    def link_cost(self, link, km: float, hours: float) -> tuple[float, float]:
        kg, h = min((c.link_cost(link, km, hours) for c in self.costs), key=lambda kg_h: kg_h[0] + self.per_hour * kg_h[1])
        return kg + self.per_hour * h, h

    def segments(self, r: SearchResult, clock) -> tuple[np.ndarray, np.ndarray]:
        """Fuel, kg, and hours of each segment of r at every speed, setting out at clock (per
        segment); shape (speeds, segments). Links at the calm-water rate."""
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        kg = seg_km * np.array([c.calm_kg_per_km for c in self.costs])[:, None]
        h = seg_km / np.array([c.speed * 3.6 for c in self.costs])[:, None]
        a = np.nonzero(~r.on_link)[0]
        kg[:, a], h[:, a] = self.kg_hours(r.lat[a], r.lon[a], r.lat[a + 1], r.lon[a + 1], np.asarray(clock)[a])
        return kg, h

    def speed_index(self, r: SearchResult) -> np.ndarray:
        """Per segment, the slowest speed that sails it in no more hours than r's clock gives it:
        on a path found with edge_cost, the speed taken."""
        _, h = self.segments(r, r.clock[:-1])
        return np.minimum((h > np.diff(r.clock) * (1 + 1e-9) + 1e-12).sum(axis=0), len(self.costs) - 1)

    def speeds_kn(self, r: SearchResult) -> np.ndarray:
        """The speed through the water of each segment of a path found with edge_cost, knots."""
        return np.array([c.speed for c in self.costs])[self.speed_index(r)] / KNOT

    def path(self, r: SearchResult) -> float:
        """Fuel, kg, along a path found with edge_cost, each segment at its own speed and time."""
        kg, _ = self.segments(r, r.clock[:-1])
        return float(kg[self.speed_index(r), np.arange(kg.shape[1])].sum())

    def sail(self, r: SearchResult, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sail r's segments one after another at the speeds of index k, from r's first clock: the
        clock at each point, and the fuel and hours of each segment at every speed, setting out then."""
        if not any(c.currents for c in self.costs):  # the hours do not depend on the weather
            seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
            clock = r.clock[0] + np.r_[0.0, np.cumsum(seg_km / (np.array([c.speed for c in self.costs])[k] * 3.6))]
            return clock, *self.segments(r, clock[:-1])
        n = len(r.lat) - 1
        clock, kg, h = np.empty(n + 1), np.empty((len(self.costs), n)), np.empty((len(self.costs), n))
        clock[0] = r.clock[0]
        for i in range(n):
            one = dataclasses.replace(r, lat=r.lat[i : i + 2], lon=r.lon[i : i + 2], on_link=r.on_link[i : i + 1])
            kg[:, i : i + 1], h[:, i : i + 1] = self.segments(one, clock[i : i + 1])
            clock[i + 1] = clock[i] + h[k[i], i]
        return clock, kg, h

    def snap_up(self, r: SearchResult) -> SearchResult:
        """Each segment at the slowest offered speed that takes no longer than it did, so the path
        arrives no later: a path smoothed after the search has segments sailed at no offered speed."""
        return dataclasses.replace(r, clock=self.sail(r, self.speed_index(r))[0])

    def shortcut(self, land: Land, r: SearchResult, keep_points=(), piece_km: float = 20.0, step: float = 1.5,
                 open_sea: OpenWater | None = None) -> SearchResult:
        """Cut the grid's steps out of a path found with edge_cost, keeping the way the weather chose.

        The cells only join in eight directions, so a found path runs in steps even across open
        sea. From each kept point, go straight to the farthest later point (of a few, each about
        `step` times farther along the path) whose great-circle arc misses land and, sailed at one
        offered speed in the weather the path met there, arrives no later than the path did and
        burns no more fuel. An arc through weather the path went round costs more and is not
        taken. Arcs are cut into pieces of piece_km at most, each timed as the path's stretch
        is, in proportion to distance. Links, the two ends and the points at keep_points stay.
        Given open_sea (quadtree.open_water), arcs it shows to miss land are not looked up: the same answers, faster.
        """
        n = len(r.lat) - 1
        lat, lon = r.lat, np.degrees(np.unwrap(np.radians(r.lon)))
        k = self.speed_index(r)
        kg, _ = self.segments(r, r.clock[:-1])
        burnt = np.r_[0.0, np.cumsum(kg[k, np.arange(n)])]
        fixed = np.zeros(n + 1, bool)
        fixed[[0, -1, *keep_points]] = True
        fixed[:-1] |= r.on_link
        fixed[1:] |= r.on_link
        pts, on_link, clock = [(lat[0], lon[0])], [], [r.clock[0]]
        i = 0
        while i < n:
            j = i + 1
            if not r.on_link[i]:
                stop = i + 1 + int(np.argmax(fixed[i + 1 :]))
                cand = np.unique(np.r_[np.minimum(i + np.ceil(step ** np.arange(64)).astype(int), stop), stop])
                cand = cand[~arcs_on_land(land, lat[i], lon[i], lat[cand], lon[cand], open_sea=open_sea)][::-1]
                for c in cand[cand > i + 1]:
                    km = float(haversine_km(lat[i], lon[i], lat[c], lon[c]))
                    f = np.arange(int(np.ceil(km / piece_km)) + 1) / int(np.ceil(km / piece_km))
                    p_lat, p_lon = along(*np.broadcast_arrays(lat[i], lon[i], lat[c], lon[c], f))
                    p_lon = lon[i] + (p_lon - lon[i] + 180) % 360 - 180
                    t = r.clock[i] + f * (r.clock[c] - r.clock[i])
                    kg_s, h_s = self.kg_hours(p_lat[:-1], p_lon[:-1], p_lat[1:], p_lon[1:], t[:-1])
                    ok = h_s.sum(axis=1) <= (r.clock[c] - r.clock[i]) * (1 + 1e-9)
                    if ok.any() and kg_s.sum(axis=1)[ok].min() <= burnt[c] - burnt[i]:
                        pts += zip(p_lat[1:], p_lon[1:])
                        on_link += [False] * (len(f) - 1)
                        clock += list(t[1:])
                        j = None
                        i = c
                        break
            if j is not None:
                pts.append((lat[j], lon[j]))
                on_link.append(bool(r.on_link[i]))
                clock.append(r.clock[j])
                i = j
        p_lat, p_lon = np.array(pts).T
        p_lon = (p_lon + 180) % 360 - 180
        km = float(haversine_km(p_lat[:-1], p_lon[:-1], p_lat[1:], p_lon[1:]).sum())
        return dataclasses.replace(r, lat=p_lat, lon=p_lon, on_link=np.array(on_link, bool), km=km, clock=np.array(clock))

    def relax(self, land: Land, r: SearchResult, keep_points=(), step_km: float = 16.0, min_km: float = 0.25,
              max_sweeps: int = 400, open_sea: OpenWater | None = None) -> SearchResult:
        """Move a path's points sideways where that is cheaper, off the grid's eight directions.

        The cells only join in eight directions, and cutting corners (shortcut) keeps to the way
        the found path went, so a path that should curve across open sea stays some way off its
        best line. Each point in turn is moved step_km to either side of the line between its
        neighbours; the move is kept if the two segments it joins, at their speeds and times,
        then arrive no later and burn no more fuel, one of them strictly, and miss land. Odd and
        even points take turns, so each move is judged against fixed neighbours. When a sweep
        keeps nothing the step is halved, down to min_km. Links, the two ends and keep_points stay.
        open_sea as in shortcut.
        """
        n = len(r.lat) - 1
        if n < 2:
            return r
        k = self.speed_index(r)
        fixed = np.zeros(n + 1, bool)
        fixed[[0, -1, *keep_points]] = True
        fixed[:-1] |= r.on_link
        fixed[1:] |= r.on_link
        lat, lon, clock = r.lat.copy(), r.lon.copy(), r.clock.copy()

        def unit(la, lo):
            p, q = np.radians(la), np.radians(lo)
            return np.stack([np.cos(p) * np.cos(q), np.cos(p) * np.sin(q), np.sin(p)])

        def pair(la, lo, i):
            """Fuel and hours of segments i-1 and i, through the points (la, lo) in place of point i."""
            kg1, h1 = self.kg_hours_at(lat[i - 1], lon[i - 1], la, lo, clock[i - 1], k[i - 1])
            kg2, h2 = self.kg_hours_at(la, lo, lat[i + 1], lon[i + 1], clock[i - 1] + h1, k[i])
            return kg1 + kg2, h1 + h2

        step, sweeps = step_km, 0
        while step >= min_km and sweeps < max_sweeps:
            sweeps += 1
            moved = 0
            for parity in (1, 2):
                i = np.arange(parity, n, 2)
                i = i[~fixed[i]]
                if not len(i):
                    continue
                kg0, h0 = pair(lat[i], lon[i], i)
                a, b, p = unit(lat[i - 1], lon[i - 1]), unit(lat[i + 1], lon[i + 1]), unit(lat[i], lon[i])
                side = np.cross(a, b, axis=0)
                side /= np.maximum(np.linalg.norm(side, axis=0), 1e-15)
                best = np.zeros(len(i), bool)
                for sign in (1.0, -1.0):
                    d = sign * step / 6371.0088
                    q = np.cos(d) * p + np.sin(d) * side
                    q_lat = np.degrees(np.arctan2(q[2], np.hypot(q[0], q[1])))
                    q_lon = np.degrees(np.arctan2(q[1], q[0]))
                    q_lon = lon[i] + (q_lon - lon[i] + 180) % 360 - 180
                    kg1, h1 = pair(q_lat, q_lon, i)
                    better = ((kg1 <= kg0) & (h1 <= h0 * (1 + 1e-12)) & ((kg1 < kg0 * (1 - 1e-9)) | (h1 < h0 * (1 - 1e-9))) & ~best
                              & np.isfinite(kg1))
                    if better.any():
                        j = np.nonzero(better)[0]
                        ok = ~(arcs_on_land(land, lat[i[j] - 1], lon[i[j] - 1], q_lat[j], q_lon[j], open_sea=open_sea)
                               | arcs_on_land(land, q_lat[j], q_lon[j], lat[i[j] + 1], lon[i[j] + 1], open_sea=open_sea))
                        j = j[ok]
                        lat[i[j]], lon[i[j]] = q_lat[j], q_lon[j]
                        kg0[j], h0[j] = kg1[j], h1[j]
                        best[j] = True
                moved += int(best.sum())
                h = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.array([c.speed * 3.6 for c in self.costs])[k]
                a = np.nonzero(~r.on_link)[0]
                h[a] = self.kg_hours_at(lat[a], lon[a], lat[a + 1], lon[a + 1], clock[a], k[a])[1]  # segments(...)[1][k, :]
                clock = clock[0] + np.r_[0.0, np.cumsum(h)]
            if moved == 0:
                step /= 2
        lon = (lon + 180) % 360 - 180
        out = dataclasses.replace(r, lat=lat, lon=lon, km=float(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()))
        clock, kg, _ = self.sail(out, k)
        return dataclasses.replace(out, clock=clock, cost=float(kg[k, np.arange(n)].sum()))

    def retime(self, r: SearchResult, eta_hours: float, dt_hours: float = 0.02, max_current: float = 2.5,
               per_hour: float | None = None) -> SearchResult:
        """The speeds on r's way that burn the least fuel arriving within eta_hours of r's first clock.

        The way is kept; each segment takes one of the offered speeds. The segments are sailed
        in order, each label (time, fuel) going on at every speed; labels that can no longer
        arrive in time, even at the fastest speed with max_current behind, are dropped. Of the
        labels whose times fall in the same bucket of dt_hours only one is kept: the least
        fuel + per_hour x hours (kg/h; by default self.per_hour, the price meet_eta left), so a
        label a little later is not kept over an earlier one for less fuel than the time is
        worth. Exact but for the buckets; r itself is returned if it arrives in time burning
        less. (Keeping every label in a bucket that is earlier or burns less, as route.exact
        does, grows without bound on a long way: slower is later and cheaper, so the labels in
        a bucket rarely dominate.)"""
        n = len(r.lat) - 1
        seg_km = haversine_km(r.lat[:-1], r.lon[:-1], r.lat[1:], r.lon[1:])
        speed = np.array([c.speed for c in self.costs])
        calm = np.array([c.calm_kg_per_km for c in self.costs])
        left_h = np.r_[np.cumsum(seg_km[::-1])[::-1], 0.0] / ((speed.max() + max_current) * 3.6)
        end = r.clock[0] + eta_hours * (1 + 1e-9)
        price = self.per_hour if per_hour is None else per_hour
        if self._args is None:
            self._args = stacked_tables(self.costs)
        counts = np.zeros((len(speed), 3), np.int64)
        t, f = np.array([float(r.clock[0])]), np.array([0.0])
        layers = []  # per segment: parent label and speed index of each label kept
        for i in range(n):
            if r.on_link[i]:
                kg = np.repeat((seg_km[i] * calm)[:, None], len(t), axis=1)
                h = np.repeat((seg_km[i] / (speed * 3.6))[:, None], len(t), axis=1)
            else:
                kg, h = np.empty((len(speed), len(t))), np.empty((len(speed), len(t)))
                kernel.fuel_kg_times(float(r.lat[i]), float(r.lon[i]), float(r.lat[i + 1]), float(r.lon[i + 1]), t, kg, h, counts, *self._args)
            t, f, parent, ks = kernel.prune(t, f, kg, h, float(left_h[i + 1]), float(end), float(dt_hours), float(price))
            if not len(t):
                raise ValueError(f"no speeds on this way arrive within {eta_hours:g} h")
            layers.append((parent, ks))
        for c, m in zip(self.costs, counts):
            c.no_waves += int(m[0])
            c.over_mcr += int(m[1])
            c.beyond_forecast += int(m[2])
        best = int(np.argmin(f))
        k = np.empty(n, int)
        for i in range(n - 1, -1, -1):
            parent, ks = layers[i]
            k[i] = ks[best]
            best = parent[best]
        clock, kg, _ = self.sail(r, k)
        out = dataclasses.replace(r, clock=clock, cost=float(kg[k, np.arange(n)].sum()))
        if r.clock[-1] <= end and (own := self.path(r)) < out.cost:  # r's own speeds, lost to the buckets, were better
            return dataclasses.replace(r, cost=own)
        return out

    def fill_slack(self, r: SearchResult, eta_hours: float, rounds: int = 20) -> SearchResult:
        """Spend the time a found path leaves before eta_hours: step segments down to the next
        slower speed, those saving the most fuel per extra hour first. Each round re-sails the
        whole path at its new times. With the current, slowing down shifts the later segments in
        time, so they may take longer than foreseen: if the path then arrives late or saves no
        fuel, the round is tried again with half as many segments, down to none."""
        seg = np.arange(len(r.lat) - 1)
        k = self.speed_index(r)
        clock, kg, h = self.sail(r, k)
        fuel = kg[k, seg].sum()
        for _ in range(rounds):
            slower = np.maximum(k - 1, 0)
            extra = h[slower, seg] - h[k, seg]
            saving = kg[k, seg] - kg[slower, seg]
            worth = np.nonzero((k > 0) & (saving > 0) & (extra > 0))[0]
            worth = worth[np.argsort(-saving[worth] / extra[worth])]
            take = worth[np.cumsum(extra[worth]) <= eta_hours - (clock[-1] - clock[0])]
            while len(take):
                k2 = k.copy()
                k2[take] -= 1
                clock2, kg2, h2 = self.sail(r, k2)
                fuel2 = kg2[k2, seg].sum()
                if fuel2 < fuel and clock2[-1] - clock2[0] <= eta_hours * (1 + 1e-9):
                    break
                take = take[: len(take) // 2]
            if len(take) == 0:
                break
            k, clock, kg, h, fuel = k2, clock2, kg2, h2, fuel2
        return dataclasses.replace(r, clock=clock, cost=float(fuel))


def meet_eta(graph: Graph, s: int, g: int, choice: SpeedChoice, eta_hours: float, tolerance: float = 0.002,
             max_tries: int = 12, max_seconds: float = math.inf, via=(), price: float | None = None,
             found: list | None = None) -> tuple[SearchResult | None, list[dict]]:
    """The least-fuel route found arriving within eta_hours, narrowing down the price of an hour.

    Through the nodes via in order, the legs are searched one after another at the same
    price, each setting out when the last one arrives.

    The first price tried is 1000 kg/h, raised fourfold until a route arrives in time; given
    `price` (one that met a like ETA before, say the last plan of the same voyage), 5% either
    side of it. Between a price arriving late and one in time, the next is read off the
    straight line through their arrival times, kept at least a tenth of the way from each.

    Stops when a route arrives within a fraction `tolerance` of the voyage before the ETA, or after max_tries
    searches. The best route that arrived in time then spends the time it has left
    (SpeedChoice.fill_slack). Returns that route (None if none arrived in time) and a log
    of the tries. Given a list `found`, every route searched, in time or late, is appended to
    it as found (its cost the fuel alone).
    """
    tries, best, hours = [], None, {}

    def search(per_hour):
        choice.per_hour = per_hour
        legs, clock = [], 0.0
        for a, b in zip([s, *via], [*via, g]):
            legs.append(choice.search(graph, a, b, max_seconds, clock))
            clock = float(legs[-1].clock[-1])
        r = join_legs(legs)
        fuel = r.cost - per_hour * float(r.clock[-1])
        hours[per_hour] = float(r.clock[-1])
        if found is not None:
            found.append(dataclasses.replace(r, cost=fuel))
        tries.append({"per_hour": round(per_hour, 1), "hours": round(float(r.clock[-1]), 1), "fuel_t": round(fuel / 1000, 2),
                      "seconds": round(r.seconds, 1), "in_time": bool(in_time(r))})
        print(f"  price {per_hour:8.1f} kg/h: {r.clock[-1]:7.1f} h, {fuel / 1000:8.2f} t, {r.seconds:5.1f} s", flush=True)
        return r, fuel

    def in_time(r):
        return r.clock[-1] <= eta_hours * (1 + 1e-9)  # a route sailed at exactly the speed for the ETA, give or take rounding

    lo, hi, grow = (0.0, 1000.0, 4.0) if price is None else (price / 1.05, price * 1.05, 1.25)
    while len(tries) < max_tries:
        r, fuel = search(hi)
        if in_time(r):
            best = (r, fuel)
            break
        lo, hi = hi, hi * grow
    if best is not None and lo not in hours and len(tries) < max_tries:  # in time at the first price: perhaps at lo too (even the slowest)
        r, fuel = search(lo)
        if in_time(r):
            best = (r, fuel)
            lo, hi = 0.0, lo
    while best is not None and len(tries) < max_tries and hi > 0.0 and best[0].clock[-1] < eta_hours * (1 - tolerance):
        mid = (lo + hi) / 2
        if lo in hours and hours[lo] > hours[hi]:
            x = (hours[lo] - eta_hours * (1 - tolerance / 2)) / (hours[lo] - hours[hi])
            mid = lo + (hi - lo) * min(max(x, 0.1), 0.9)
        r, fuel = search(mid)
        if in_time(r):
            hi = mid
            if fuel < best[1]:
                best = (r, fuel)
        else:
            lo = mid
    if best is None:
        return None, tries
    r, fuel = best
    return choice.fill_slack(dataclasses.replace(r, cost=fuel), eta_hours), tries


def run(graph: Graph, ship, forecast, speeds_kn, fixed_kn: float, max_tries: int, max_seconds: float, tables: str | None = None) -> list[dict]:
    """For each test voyage, the moving-weather route at a fixed speed against the least-fuel one
    choosing speeds that arrives no later. Wave tables are kept in the folder `tables` if given."""
    from shiprouting.fuel.waves import AddedResistanceTable
    from shiprouting.sea.latlon import ROUTES

    departure = forecast.time[0]
    costs = {}
    for kn in sorted({*speeds_kn, fixed_kn}):
        t0 = time.perf_counter()
        if tables:
            waves = AddedResistanceTable.cached(tables, ship.seakeeping, kn * KNOT)
        else:
            waves = AddedResistanceTable.build(ship.seakeeping, kn * KNOT)
        costs[kn] = FuelCost(ship, forecast, departure, kn * KNOT, waves, frozen=False)
        print(f"{kn:g} kn: wave added resistance table ready in {time.perf_counter() - t0:.0f} s", flush=True)
    fixed, choice = costs[fixed_kn], SpeedChoice([costs[kn] for kn in sorted(speeds_kn)])
    rows = []
    for name, (start, goal) in ROUTES.items():
        s, g = graph.nearest(*start), graph.nearest(*goal)
        row = {"route": name, "departure": str(departure)}
        print(name, flush=True)
        found = {"fixed": astar(graph, s, g, fixed.costs(graph), fixed.link_cost, lambda n: 0.0)}
        eta = float(found["fixed"].clock[-1])
        t0 = time.perf_counter()
        r, row["tries"] = meet_eta(graph, s, g, choice, eta, max_tries=max_tries, max_seconds=max_seconds)
        row["eta_seconds"] = round(time.perf_counter() - t0, 1)
        if r is not None:
            found["eta"] = r
        for key, r in found.items():
            kn = choice.speeds_kn(r)
            row[key] = {
                "km": round(r.km),
                "hours": round(float(r.clock[-1]), 1),
                "fuel_t": round(r.cost / 1000, 2),
                "speeds_kn": {f"{v:g}": int(n) for v, n in zip(*np.unique(np.round(kn, 3), return_counts=True))},
                "expanded": r.expanded,
                "seconds": round(r.seconds, 1),
                "links": r.links,
                "lat": np.round(r.lat, 6).tolist(),
                "lon": np.round(r.lon, 6).tolist(),
                "clock": np.round(r.clock, 2).tolist(),
            }
        print({k: ({kk: vv for kk, vv in v.items() if kk not in ("lat", "lon", "clock")} if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    import glob
    import json

    from shiprouting.fuel.ships import KCS_SHIP
    from shiprouting.sea.land import load_osm_land_polygons
    from shiprouting.weather import ecmwf

    parser = argparse.ArgumentParser(description="Least-fuel routes and speeds arriving no later than a fixed speed, in moving weather.")
    parser.add_argument("land", help="land polygons shapefile")
    parser.add_argument("forecast", help="glob of the ECMWF GRIB2 files")
    parser.add_argument("out", help="results .json")
    parser.add_argument("--min-step", type=float, default=1 / 64)
    parser.add_argument("--links", help="folder with the Overpass canal and strait files (see links.LINKS)")
    parser.add_argument("--speeds-kn", default="12,14,16,18,20", help="speeds to choose from, comma-separated")
    parser.add_argument("--fixed-kn", type=float, default=16.0, help="the fixed speed to compare with")
    parser.add_argument("--max-tries", type=int, default=12, help="searches per voyage to meet the arrival time")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="give up a search after this long")
    parser.add_argument("--tables", help="folder to keep the wave added resistance tables in between runs")
    args = parser.parse_args()

    land = load_osm_land_polygons(args.land)
    links = []
    if args.links:
        from shiprouting.sea.links import load_links

        links = load_links(args.links)
    t0 = time.perf_counter()
    graph = Graph.build(land, min_step=args.min_step, links=links)
    print(f"graph: {len(graph.lat)} cells, built in {time.perf_counter() - t0:.0f} s", flush=True)
    forecast = ecmwf.load(sorted(glob.glob(args.forecast)))
    speeds = [float(v) for v in args.speeds_kn.split(",")]
    rows = run(graph, KCS_SHIP, forecast, speeds, args.fixed_kn, args.max_tries, args.max_seconds, args.tables)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
