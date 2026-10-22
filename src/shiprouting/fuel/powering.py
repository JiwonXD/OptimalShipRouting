"""From hull resistance to brake power."""

from dataclasses import dataclass

from shiprouting.fuel.holtrop import RHO_SEA
from shiprouting.fuel.propeller import OpenWaterCurve, delivered_power, operating_point


@dataclass(frozen=True)
class PropulsionFactors:
    t: float  # thrust deduction fraction
    w: float  # effective wake fraction
    eta_R: float  # relative rotative efficiency
    eta_S: float  # shaft transmission efficiency


def brake_power(
    resistance: float,
    speed: float,
    factors: PropulsionFactors,
    curve: OpenWaterCurve,
    D: float,
    rho: float = RHO_SEA,
) -> float:
    """Brake power, W, to overcome ``resistance`` (N) at ``speed`` (m/s)."""
    thrust = resistance / (1 - factors.t)
    point = operating_point(curve, thrust, speed * (1 - factors.w), D, rho)
    return delivered_power(point, D, rho, factors.eta_R) / factors.eta_S
