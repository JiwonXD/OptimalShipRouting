"""How much the fuel model's uncertain inputs move a result: Monte Carlo and Sobol indices.

Each uncertain input of the ship model gets a range (PARAMETERS, docs/experiment-plan.md); a
sample is one value of each, drawn uniformly. A sample changes the ship (calm-water resistance,
propulsion factors, the engine's part-load fuel use) and scales the air and wave added
resistance. Tracks are judged on the moves' resistance parts (FuelCost.resistance_parts) at
their speeds through the water, which do not change with the ship, so a sample costs no
weather lookups: only a fuel rate table per speed (rate_tables).

Sobol indices (Saltelli 2010 sampling, Jansen estimators): S1, the share of the result's
variance a parameter causes alone; ST, its share including its interactions with the others.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from shiprouting.fuel.powering import brake_power
from shiprouting.fuel.ship import Ship


@dataclass(frozen=True)
class Parameter:
    name: str
    low: float
    high: float
    nominal: float
    why: str


PARAMETERS = (
    Parameter("calm_resistance", 0.95, 1.15, 1.0,
              "x calm-water resistance: model-to-full-scale extrapolation about ±5 %, hull fouling up to +10 %"),
    Parameter("thrust_deduction", 0.14, 0.22, 0.180, "t: not published for KCS; Holtrop regression 0.18, ±0.04"),
    Parameter("wake", 0.15, 0.209, 0.209, "w: model-scale value from case 2.5; full-scale wake is smaller"),
    Parameter("eta_R", 0.98, 1.03, 1.008, "relative rotative efficiency, ±2.5 % around the case 2.5 value"),
    Parameter("air", 0.65, 1.35, 1.0,
              "x air resistance: A_XV assumed (600-1000 m² around 800) and C_DA from another container ship (±20 %)"),
    Parameter("waves", 0.7, 1.3, 1.0, "x wave added resistance: SNNM scatter against experiments in oblique and short waves"),
    Parameter("sfoc_part_load", -5.0, 5.0, 0.0,
              "g/kWh added to SFOC at 50 % load (75 % and 100 % kept): engine tuning moves the part-load point"),
)  # fmt: skip
NOMINAL = np.array([p.nominal for p in PARAMETERS])


def ship_with(ship: Ship, p) -> Ship:
    """ship with one sample of PARAMETERS (in their order) applied, but for air and waves, which
    scale the resistance parts outside the ship."""
    calm, t, w, eta_R, _, _, dsfoc = p
    (x0, y0), *rest = ship.engine.sfoc_points
    return dataclasses.replace(
        ship,
        resistance=lambda speed, r=ship.resistance: calm * r(speed),
        factors=dataclasses.replace(ship.factors, t=t, w=w, eta_R=eta_R),
        engine=dataclasses.replace(ship.engine, sfoc_points=((x0, y0 + dsfoc), *rest)),
    )


def rate_tables(ship: Ship, speeds, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per speed (m/s): the calm-water resistance, and the fuel rate, kg/h, on n total resistances
    from 0.2 x calm up to the engine's MCR (as FuelCost tabulates it). Shapes (speeds,), (speeds, n)."""
    calm, R, kg_h = [], [], []
    for v in speeds:

        def power(r, v=v):
            return brake_power(r, v, ship.factors, ship.propeller, ship.propeller_D)

        r0 = ship.resistance(v)
        lo, hi = 0.2 * r0, r0
        while power(hi) < ship.engine.mcr:
            hi *= 2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if power(mid) < ship.engine.mcr else (lo, mid)
        grid = np.linspace(0.2 * r0, lo, n)
        calm.append(r0)
        R.append(grid)
        kg_h.append([ship.engine.fuel_rate(power(r)) for r in grid])
    return np.array(calm), np.array(R), np.array(kg_h)


def track_fuel(speed_index, air, wave, h, tables, air_scale: float, wave_scale: float) -> tuple[float, int]:
    """Fuel, kg, of a track's moves at their speeds (indices into the tables' speeds), and how
    many of them need more than MCR (taken at MCR's fuel rate: the speed loss is not modelled)."""
    calm, R_grid, kg_h = tables
    R = calm[speed_index] + air_scale * air + wave_scale * wave
    rate = np.empty(len(R))
    for k in np.unique(speed_index):
        on = speed_index == k
        rate[on] = np.interp(R[on], R_grid[k], kg_h[k])
    return float((rate * h).sum()), int((R > R_grid[speed_index, -1]).sum())


def saltelli(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples A, B (n x d) and AB (d x n x d: A with column i from B), uniform in PARAMETERS' ranges."""
    low, high = np.array([p.low for p in PARAMETERS]), np.array([p.high for p in PARAMETERS])
    d = len(PARAMETERS)
    A, B = (low + (high - low) * rng.random((n, d)) for _ in range(2))
    AB = np.repeat(A[None], d, axis=0)
    for i in range(d):
        AB[i, :, i] = B[:, i]
    return A, B, AB


def sobol(fA, fB, fAB) -> tuple[np.ndarray, np.ndarray]:
    """First-order and total indices from the results on A, B and AB (Saltelli 2010, Jansen 1999)."""
    var = np.var(np.r_[fA, fB])
    S1 = np.mean(fB[None] * (fAB - fA[None]), axis=1) / var
    ST = 0.5 * np.mean((fA[None] - fAB) ** 2, axis=1) / var
    return S1, ST
