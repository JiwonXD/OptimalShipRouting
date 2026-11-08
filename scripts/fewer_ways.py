"""How much a plan would lose by searching fewer prices or finishing fewer of the ways found.

Reads voyage logs written with the "found" block (plan.py's kept_t, ways_t, ways_same_as,
shortest_finished_t) and, for every plan, picks again among the ways it weighed as a cheaper
plan would have:

- tries N: only the first N price searches (meet_eta's max_tries), the ways they found finished;
- top K: every price searched, only the K ways cheapest on the cells finished;
- in time: only the ways that arrived in time on the cells finished.

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
    grid, in_time, done = candidates(found)
    n = len(done)
    out = {"all": pick(found, range(n))}
    for k in (2, 3, 4, 6, 8):
        if k < n:
            # with the first k tries only, the kept way is the cheapest of them in time (finished)
            kept = [done[i] for i in range(k) if in_time[i] and done[i] is not None]
            first = dict(found, kept_t=min(kept) if kept else found["kept_t"])
            out[f"tries {k}"] = pick(first, range(k))
        else:
            out[f"tries {k}"] = out["all"]
    order = sorted(range(n), key=lambda i: grid[i])
    for k in (1, 2, 3, 4):
        out[f"top {k}"] = pick(found, order[:k])
    out["in time"] = pick(found, [i for i in range(n) if in_time[i]])
    out["n_tries"], out["n_finished"] = n, sum(1 for i in range(n) if (found["ways_same_as"] or [None] * n)[i] is None)
    return out


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
                p = policies(found)
                rows.append({"voyage": os.path.basename(path)[:-5], "plan": k, "planned_t": entry["planned_t"],
                             "seconds": entry["seconds"]} | p)
    if not rows:
        raise SystemExit("no voyage logs with a 'found' block")
    names = [k for k in rows[0] if k not in ("voyage", "plan", "planned_t", "seconds", "n_tries", "n_finished")]
    print(f"{len(rows)} plans; model check: 'all' vs planned_t, max |diff| "
          f"{max(abs(r['all'] - r['planned_t']) for r in rows if r['all'] is not None):.2f} t")
    print(f"{'policy':10s} {'plans worse':>11s} {'sum +t':>8s} {'worst +t':>8s} {'worst %':>8s}")
    for name in names:
        d = [(r[name] - r["all"], (r[name] - r["all"]) / r["all"]) for r in rows if r[name] is not None and r["all"] is not None]
        worse = [x for x in d if x[0] > 0.005]
        print(f"{name:10s} {len(worse):11d} {sum(x[0] for x in d):8.2f} {max((x[0] for x in d), default=0):8.2f}"
              f" {100 * max((x[1] for x in d), default=0):7.2f}%")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
