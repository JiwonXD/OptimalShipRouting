"""Propeller open-water characteristics and operating point.

The open-water curve is used as measured at model scale; the ITTC-1978
scale correction of K_T and K_Q (a few percent at most) is not applied.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenWaterCurve:
    """K_T and K_Q against advance ratio J, ascending in J."""

    J: tuple[float, ...]
    K_T: tuple[float, ...]
    K_Q: tuple[float, ...]

    def k_t(self, J: float) -> float:
        return _interp(self.J, self.K_T, J)

    def k_q(self, J: float) -> float:
        return _interp(self.J, self.K_Q, J)


def _interp(xs: tuple[float, ...], ys: tuple[float, ...], x: float) -> float:
    if not xs[0] <= x <= xs[-1]:
        raise ValueError(f"J={x:.3f} is outside the curve {xs[0]}..{xs[-1]}")
    for i in range(len(xs) - 1):
        if x <= xs[i + 1]:
            w = (x - xs[i]) / (xs[i + 1] - xs[i])
            return (1 - w) * ys[i] + w * ys[i + 1]
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class OperatingPoint:
    J: float
    n: float  # revolutions per second
    K_T: float
    K_Q: float


def operating_point(
    curve: OpenWaterCurve, thrust: float, advance_speed: float, D: float, rho: float
) -> OperatingPoint:
    """Find J where the propeller gives ``thrust`` (N) at ``advance_speed`` (m/s).

    Solves K_T(J) = c J² with c = T / (ρ D² V_A²). K_T falls and c J² rises with
    J, so the root is unique; it is found by bisection.
    """
    if thrust <= 0 or advance_speed <= 0:
        raise ValueError("thrust and advance speed must be positive")
    c = thrust / (rho * D**2 * advance_speed**2)
    lo, hi = curve.J[0], curve.J[-1]
    if curve.k_t(hi) - c * hi**2 > 0:
        raise ValueError("thrust is below what the curve covers")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if curve.k_t(mid) - c * mid**2 > 0:
            lo = mid
        else:
            hi = mid
    J = 0.5 * (lo + hi)
    return OperatingPoint(
        J=J, n=advance_speed / (J * D), K_T=curve.k_t(J), K_Q=curve.k_q(J)
    )


def delivered_power(point: OperatingPoint, D: float, rho: float, eta_R: float) -> float:
    """Power delivered to the propeller behind the hull, W."""
    torque = point.K_Q * rho * point.n**2 * D**5 / eta_R
    return 2 * math.pi * point.n * torque
