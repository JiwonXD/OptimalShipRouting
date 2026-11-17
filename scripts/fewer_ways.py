"""How much a plan would lose by searching fewer prices or finishing fewer of the ways found.

Reads voyage logs written with the "found" block (plan.py's kept_t, ways_t, ways_same_as,
shortest_finished_t) and, for every plan, picks again among the ways it weighed as a cheaper
plan would have:

- tries N: only the first N price searches (meet_eta's max_tries), the ways they found finished;
- top K: every price searched, only the K ways cheapest on the cells finished;
- in time: only the ways that arrived in time on the cells finished;
- and the first N tries with only their ways in time finished.

Each also with a rough estimate of the seconds it saves (seconds_saved).

The shortest route as it sails, and with its speeds chosen again before and after it is moved
sideways, are weighed in every case. A
way a cheaper plan would have kept but this one did not finish (a repeat of the kept one) is
taken at the kept one's fuel. The price that chooses the speeds again is the full plan's, so
with fewer tries it is only nearly what that plan would use.

    python scripts/fewer_ways.py FOLDER [FOLDER ...] [--out fewer.csv]
"""

import argparse
import csv
import glob
import json
import os


def candidates(found: dict) -> tuple[list, list, list]:
    """Per try: the way's fuel on the cells, in time?, and finished (None: could not arrive in time)."""
    tries = [t for leg in found["tries"] for t in leg]
    ways, same = found["ways_t"] or [], found["ways_same_as"] or []
    done = []
    for i in range(len(ways)):
        s = same[i]
        if s is None:
            done.append(ways[i])
        elif s == "kept":
            done.append(found["kept_t"])
        else:
            done.append(done[s])
    return [t["fuel_t"] for t in tries], [t["in_time"] for t in tries], done


def pick(found: dict, use) -> float | None:
    """The cheapest of the kept way, the ways at indices `use` and the shortest route's two."""
    _, _, done = candidates(found)
    pool = [found["kept_t"]] + [done[i] for i in use] + list(found["shortest_finished_t"] or []) + [found.get("shortest_t")]
    pool = [v for v in pool if v is not None]
    return min(pool) if pool else None


def policies(found: dict) -> dict:
    """Per policy: (fuel of the way kept, try indices searched, way indices finished)."""
    grid, in_time, done = candidates(found)
    n = len(done)
    every = range(n)
    out = {"all": (pick(found, every), every, every)}

    def first(k, use):
        # with the first k tries only, the kept way is the cheapest of them in time (finished)
        kept = [done[i] for i in range(k) if in_time[i] and done[i] is not None]
        return pick(dict(found, kept_t=min(kept) if kept else found["kept_t"]), use)

    for k in (2, 3, 4, 6, 8):
        m = min(k, n)
        out[f"tries {k}"] = (first(m, range(m)), range(m), range(m))
        timely = [i for i in range(m) if in_time[i]]
        out[f"tries {k} + in time"] = (first(m, timely), range(m), timely)
    order = sorted(every, key=lambda i: grid[i])
    for k in (1, 2, 3, 4):
        out[f"top {k}"] = (pick(found, order[:k]), every, order[:k])
    timely = [i for i in every if in_time[i]]
    out["in time"] = (pick(found, timely), every, timely)
    return out


def seconds_saved(found: dict, searched, finished) -> float:
    """A rough estimate: the dropped tries' search seconds, and the dropped ways' share of the rest
    of the least-fuel part (the kept way and the shortest route's two retimes counted as 2.5 ways)."""
    tries = [t for leg in found["tries"] for t in leg]
    same = found["ways_same_as"] or [None] * len(tries)
    new = [i for i in range(len(tries)) if same[i] is None]  # finished (or tried to be)
    search = sum(t["seconds"] for t in tries)
    per_way = max(found["least_fuel_s"] - search, 0.0) / (len(new) + 2.5)
    return sum(t["seconds"] for i, t in enumerate(tries) if i not in set(searched)) + per_way * len(set(new) - set(finished))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = []
    for folder in args.folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            with open(path, encoding="utf-8") as f:
                v = json.load(f)
            for k, entry in enumerate(v["sailed"]["log"]):
                if "found" not in entry:
                    continue
                found = entry["found"]
                if "shortest_t" not in found and k == 0:  # logs before it was kept: the voyage's own is the first plan's
                    found = dict(found, shortest_t=v["planned"]["shortest_t"])
                for name, (fuel, searched, finished) in policies(found).items():
                    rows.append({"voyage": os.path.basename(path)[:-5], "plan": k, "planned_t": entry["planned_t"],
                                 "seconds": entry["seconds"], "policy": name, "fuel_t": fuel,
                                 "saved_s": round(seconds_saved(found, searched, finished), 1)})
    if not rows:
        raise SystemExit("no voyage logs with a 'found' block")
    full = {(r["voyage"], r["plan"]): r["fuel_t"] for r in rows if r["policy"] == "all"}
    print(f"{len(full)} plans; model check: 'all' vs planned_t, max |diff| "
          f"{max(abs(r['fuel_t'] - r['planned_t']) for r in rows if r['policy'] == 'all'):.2f} t")
    routes = sorted({r["voyage"].rsplit("_", 2)[0] for r in rows})
    print(f"{'policy':20s} {'worse':>5s} {'sum +t':>7s} {'worst +t':>8s} {'worst %':>7s}  median plan s after (saved), " + ", ".join(routes))
    for name in dict.fromkeys(r["policy"] for r in rows):
        d = [(r["fuel_t"] - full[r["voyage"], r["plan"]], (r["fuel_t"] - full[r["voyage"], r["plan"]]) / full[r["voyage"], r["plan"]])
             for r in rows if r["policy"] == name and r["fuel_t"] is not None]
        worse = [x for x in d if x[0] > 0.005]
        by_route = []
        for route in routes:
            mine = [r for r in rows if r["policy"] == name and r["voyage"].rsplit("_", 2)[0] == route]
            after = sorted(r["seconds"] - r["saved_s"] for r in mine)
            saved = sorted(r["saved_s"] for r in mine)
            by_route.append(f"{after[len(after) // 2]:.0f} ({saved[len(saved) // 2]:.0f})")
        print(f"{name:20s} {len(worse):5d} {sum(x[0] for x in d):7.2f} {max((x[0] for x in d), default=0):8.2f}"
              f" {100 * max((x[1] for x in d), default=0):6.2f}%  " + ", ".join(by_route))
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
