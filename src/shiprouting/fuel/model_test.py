"""Full-scale calm-water resistance from a towing-tank resistance test.

Extrapolation follows the ITTC-1978 method: the residuary coefficient
C_R = C_TM - (1 + k) C_FM is taken over at equal Froude number, and

C_TS = (1 + k) C_FS + C_R + ΔC_F

with ΔC_F the ITTC-1978 roughness allowance. Air resistance (C_AA) is left
out because the wind model accounts for the ship's own relative wind.
The form factor comes from the low-speed points by Prohaska's method.

Froude and Reynolds numbers both use the length ``L`` of the test.
"""

import math
from dataclasses import dataclass

from shiprouting.fuel.holtrop import G, NU_SEA, RHO_SEA, friction_coefficient


@dataclass(frozen=True)
class ResistanceTest:
    """Measured total resistance coefficients of a model."""

    L: float  # model length, m
    wetted_surface_ratio: float  # S / L²
    nu: float  # tank water kinematic viscosity, m²/s
    froude: tuple[float, ...]  # ascending
    C_T: tuple[float, ...]


def prohaska_form_factor(test: ResistanceTest, max_froude: float = 0.2) -> float:
    """1 + k from C_T/C_F = (1 + k) + c Fn⁴/C_F, least squares over Fn <= max_froude."""
    xs, ys = [], []
    for Fn, C_T in zip(test.froude, test.C_T):
        if Fn > max_froude:
            continue
        C_F = friction_coefficient(Fn * math.sqrt(G * test.L), test.L, test.nu)
        xs.append(Fn**4 / C_F)
        ys.append(C_T / C_F)
    if len(xs) < 2:
        raise ValueError("Prohaska's method needs at least two points")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return my - slope * mx


def residuary_coefficient(test: ResistanceTest, froude: float, form_factor: float) -> float:
    """C_R at ``froude``, linearly interpolated between test points."""
    Fn, C_T = test.froude, test.C_T
    if not Fn[0] <= froude <= Fn[-1]:
        raise ValueError(f"Fn={froude:.3f} is outside the test range {Fn[0]}..{Fn[-1]}")

    def C_R(i: int) -> float:
        V = Fn[i] * math.sqrt(G * test.L)
        return C_T[i] - form_factor * friction_coefficient(V, test.L, test.nu)

    for i in range(len(Fn) - 1):
        if froude <= Fn[i + 1]:
            w = (froude - Fn[i]) / (Fn[i + 1] - Fn[i])
            return (1 - w) * C_R(i) + w * C_R(i + 1)
    raise AssertionError("unreachable")


def roughness_allowance(speed: float, L: float, nu: float = NU_SEA, k_s: float = 150e-6) -> float:
    """ITTC-1978 ΔC_F for hull roughness ``k_s`` (m)."""
    Re = speed * L / nu
    return 0.044 * ((k_s / L) ** (1 / 3) - 10 * Re ** (-1 / 3)) + 0.000125


def calm_water_resistance(
    test: ResistanceTest,
    L: float,
    speed: float,
    rho: float = RHO_SEA,
    nu: float = NU_SEA,
    form_factor: float | None = None,
) -> float:
    """Full-scale calm-water resistance, N, of a ship of length ``L`` at ``speed`` (m/s)."""
    if form_factor is None:
        form_factor = prohaska_form_factor(test)
    froude = speed / math.sqrt(G * L)
    C_T = (
        form_factor * friction_coefficient(speed, L, nu)
        + residuary_coefficient(test, froude, form_factor)
        + roughness_allowance(speed, L, nu)
    )
    S = test.wetted_surface_ratio * L**2
    return 0.5 * rho * speed**2 * S * C_T
