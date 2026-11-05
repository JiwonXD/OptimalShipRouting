"""Sharp turns in found routes: points where the course turns by 60° or more."""

import numpy as np

from shiprouting.sea.latlon import haversine_km, turn_deg


def summary(lat, lon, sharp: float = 60.0) -> dict:
    """How many points of a path turn by `sharp` degrees or more, the largest turn, and where they are."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    turn = turn_deg(lat, lon)
    seg_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    at = np.nonzero(turn >= sharp)[0]  # turn[i] is at point i + 1, between segments i and i + 1
    legs = np.minimum(seg_km[at], seg_km[at + 1])
    return {
        "points": len(lat),
        "sharp": len(at),
        "over_90": int((turn >= 90).sum()),
        "max_deg": round(float(turn.max()), 1) if len(turn) else 0.0,
        "shorter_leg_km": round(float(np.median(legs)), 1) if len(at) else None,
        "where": [(round(float(lat[i + 1]), 3), round(float(lon[i + 1]), 3), round(float(turn[i]))) for i in at],
    }


def around(lat, lon, sharp: float, n: int) -> None:
    """Print the points either side of each sharp turn: position, turn there and the leg on to the next."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    turn = np.r_[np.nan, turn_deg(lat, lon), np.nan]  # at every point
    seg_km = np.r_[haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]), np.nan]
    for i in np.nonzero(turn >= sharp)[0]:
        for j in range(max(i - n, 0), min(i + n + 1, len(lat))):
            print(f"    {'>' if j == i else ' '} {j:5d} ({lat[j]:.4f}, {lon[j]:.4f}) turn {turn[j]:5.0f}°, next leg {seg_km[j]:7.2f} km")
        print()


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Count sharp turns in the routes of results .json files.")
    parser.add_argument("results", nargs="+", help="results .json files with lat/lon per route")
    parser.add_argument("--sharp", type=float, default=60.0, help="a turn this large or larger counts, degrees")
    parser.add_argument("--around", type=int, default=0, help="also print this many points either side of each sharp turn")
    args = parser.parse_args()

    for path in args.results:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        rows = [rows] if isinstance(rows, dict) else rows  # one plan, or rows of test voyages
        print(path)
        for row in rows:
            for key, found in row.items():
                if isinstance(found, dict) and "lat" in found:
                    s = summary(found["lat"], found["lon"], args.sharp)
                    print(f"  {row.get('route', '')} {key}: {s['sharp']} of {s['points']} points turn ≥{args.sharp:g}° "
                          f"({s['over_90']} ≥90°, max {s['max_deg']}°, shorter leg median {s['shorter_leg_km']} km) {s['where'][:8]}")
                    if args.around:
                        around(found["lat"], found["lon"], args.sharp, args.around)


if __name__ == "__main__":
    main()
