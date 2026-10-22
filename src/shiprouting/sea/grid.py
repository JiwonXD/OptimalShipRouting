"""Navigable sea graph on the H3 hexagonal grid.

Open sea uses cells of ``ocean_res``. Cells that touch land are split into
their children, down to ``max_res``, so coasts and straits get small cells
while the open ocean stays coarse. A cell at ``max_res`` that still touches
land is kept when its centre is at sea. Neighbouring cells (of the same or
different resolution) are joined by an edge unless the straight segment
between their centres touches land.
"""

import argparse
import logging
import time
from dataclasses import dataclass

import h3.api.basic_int as h3
import numpy as np
import shapely

from shiprouting.sea.geometry import haversine_km, path_geometry, ring_geometry
from shiprouting.sea.land import Land, load_osm_land_polygons

log = logging.getLogger(__name__)

# Straits narrower than a cell at the default max_res, refined further:
# (lon_min, lat_min, lon_max, lat_max) of the cell centres, resolution.
STRAITS = {
    "Bosporus": ((28.95, 40.98, 29.20, 41.25), 9),
    "Dardanelles": ((26.15, 39.95, 26.75, 40.50), 9),
}


@dataclass(frozen=True)
class SeaGraph:
    cells: np.ndarray  # H3 cell ids (uint64)
    lat: np.ndarray  # cell centre, degrees
    lon: np.ndarray
    edge_i: np.ndarray  # node index pairs, each edge once
    edge_j: np.ndarray
    edge_km: np.ndarray  # great-circle length

    def save(self, path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path) -> "SeaGraph":
        with np.load(path) as f:
            return cls(**{k: f[k] for k in f.files})


def build(
    land: Land, ocean_res: int = 4, max_res: int = 7, max_lat: float = 85.0, refine=tuple(STRAITS.values())
) -> SeaGraph:
    cells = sea_cells(land, ocean_res, max_res, max_lat, refine)
    lat, lon = np.array([h3.cell_to_latlng(int(c)) for c in cells]).T
    log.info("%d nodes", len(cells))
    i, j = neighbour_pairs(cells, ocean_res)
    log.info("%d neighbour pairs", len(i))
    lines = np.array(
        [path_geometry(lat[[a, b]], lon[[a, b]]) for a, b in zip(i, j)], dtype=object
    )
    keep = ~land.crosses(lines)
    i, j = i[keep], j[keep]
    log.info("%d edges clear of land", len(i))
    return SeaGraph(cells, lat, lon, i, j, haversine_km(lat[i], lon[i], lat[j], lon[j]))


def sea_cells(land: Land, ocean_res: int, max_res: int, max_lat: float, refine=tuple(STRAITS.values())) -> np.ndarray:
    """Cells of the adaptive sea tiling, from resolution 0 down."""
    nodes = []
    frontier = np.array(sorted(h3.get_res0_cells()), dtype=np.uint64)
    from_land = np.zeros(len(frontier), bool)  # parent was all land
    for res in range(0, max([max_res, *(r for _, r in refine)]) + 1):
        if not len(frontier):
            break
        lat, lon = np.array([h3.cell_to_latlng(int(c)) for c in frontier]).T
        inside = np.abs(lat) <= max_lat
        frontier, from_land, lat, lon = frontier[inside], from_land[inside], lat[inside], lon[inside]
        # A cell around a pole has no lon/lat polygon; split it unexamined.
        poles = np.isin(frontier, [h3.latlng_to_cell(90.0, 0.0, res), h3.latlng_to_cell(-90.0, 0.0, res)])
        frac = np.full(len(frontier), 0.5)
        frac[~poles] = land.fraction(cell_polygons(frontier[~poles]))
        sea = frac <= 1e-12
        all_land = frac >= 1 - 1e-12
        mixed = ~sea & ~all_land
        log.info("res %d: %d cells, %d sea, %d mixed, %d land", res, len(frontier), sea.sum(), mixed.sum(), all_land.sum())

        limit = np.full(len(frontier), max_res)
        for (x0, y0, x1, y1), r in refine:
            box = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
            limit[box] = np.maximum(limit[box], r)
        last = res >= limit
        if res >= ocean_res:
            nodes.append(frontier[sea])
            # A cell that still touches land at its finest level is kept if its centre is at sea.
            done = mixed & last
            nodes.append(frontier[done][~land.contains(lat[done], lon[done])])
            split = mixed & ~last
        else:
            split = sea | mixed
        # Children of an all-land cell can stick out into the sea, so look one
        # level further before dropping them.
        split_land = all_land & ~from_land & ~last
        parents = np.concatenate([frontier[split], frontier[split_land]])
        flags = np.concatenate([np.zeros(split.sum(), bool), np.ones(split_land.sum(), bool)])
        children = [np.array(h3.cell_to_children(int(c), res + 1), dtype=np.uint64) for c in parents]
        counts = [len(c) for c in children]
        frontier = np.concatenate(children) if children else np.array([], dtype=np.uint64)
        from_land = np.repeat(flags, counts)
    return np.sort(np.concatenate(nodes))


def cell_polygons(cells) -> np.ndarray:
    polys = []
    for c in cells:
        lat, lon = np.array(h3.cell_to_boundary(int(c))).T
        polys.append(ring_geometry(lat, lon))
    return np.array(polys, dtype=object)


def neighbour_pairs(cells: np.ndarray, ocean_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of touching cells.

    For each cell, every same-resolution neighbour that is not itself a node
    is replaced by its nearest ancestor that is. Pairs between a coarse cell
    and finer cells are found from the finer side.
    """
    index = {int(c): k for k, c in enumerate(cells)}
    pairs = set()
    for k, c in enumerate(cells):
        for n in h3.grid_disk(int(c), 1):
            a = n
            while a not in index and h3.get_resolution(a) > ocean_res:
                a = h3.cell_to_parent(a)
            m = index.get(a)
            if m is not None and m != k:
                pairs.add((min(k, m), max(k, m)))
    if not pairs:
        return np.array([], int), np.array([], int)
    i, j = np.array(sorted(pairs)).T
    return i, j


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the sea graph from OSM land polygons.")
    parser.add_argument("land", help="land-polygons-split-4326 shapefile")
    parser.add_argument("out", help="output .npz")
    parser.add_argument("--ocean-res", type=int, default=4)
    parser.add_argument("--max-res", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    land = load_osm_land_polygons(args.land)
    log.info("%d land polygons loaded", len(land.geoms))
    graph = build(land, args.ocean_res, args.max_res)
    graph.save(args.out)
    log.info("saved %s in %.0f s", args.out, time.time() - t0)


if __name__ == "__main__":
    main()
