"""Air resistance from relative wind, ITTC 7.5-04-01-01.1 (Rev 08, 2024).

R_air = ½ ρ_A C_DA(ψ_WR) A_XV V_WR²

This is the full air resistance, including the still-air part caused by the
ship's own speed, because the calm-water resistance leaves air out.
"""

import math
from dataclasses import dataclass

RHO_AIR = 1.225  # kg/m³, standard atmosphere at sea level


@dataclass(frozen=True)
class Windage:
    A_XV: float  # transverse projected area above the waterline, m²
    angles: tuple[float, ...]  # relative wind angle, degrees, 0 = head, ascending to 180
    C_DA: tuple[float, ...]  # wind resistance coefficient (= -C_X)

    def coefficient(self, angle: float) -> float:
        """C_DA at relative wind ``angle`` (rad, either side of the bow)."""
        a = math.degrees(abs(math.remainder(angle, 2 * math.pi)))
        xs, ys = self.angles, self.C_DA
        for i in range(len(xs) - 1):
            if a <= xs[i + 1]:
                w = (a - xs[i]) / (xs[i + 1] - xs[i])
                return (1 - w) * ys[i] + w * ys[i + 1]
        return ys[-1]


def relative_wind(
    wind_speed: float, wind_angle: float, ship_speed: float
) -> tuple[float, float]:
    """Relative wind (speed m/s, angle rad, 0 = head) seen from the moving ship.

    ``wind_angle`` is where the true wind comes from relative to the bow
    (0 = head wind) and ``ship_speed`` is the speed over ground.
    """
    # Air velocity relative to the ship, in ship axes (x forward, y to starboard).
    ax = -wind_speed * math.cos(wind_angle) - ship_speed
    ay = -wind_speed * math.sin(wind_angle)
    return math.hypot(ax, ay), math.atan2(-ay, -ax)


def air_resistance(
    windage: Windage,
    wind_speed: float,
    wind_angle: float,
    ship_speed: float,
    rho_air: float = RHO_AIR,
) -> float:
    """Air resistance, N, positive against the direction of travel."""
    V_WR, psi_WR = relative_wind(wind_speed, wind_angle, ship_speed)
    return 0.5 * rho_air * windage.coefficient(psi_WR) * windage.A_XV * V_WR**2
