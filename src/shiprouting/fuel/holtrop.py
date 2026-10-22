"""Calm-water resistance by Holtrop & Mennen (1982).

J. Holtrop and G.G.J. Mennen, "An approximate power prediction method",
International Shipbuilding Progress, 1982.

R_T = R_F (1 + k1) + R_APP + R_W + R_B + R_TR + R_A

Only the wave-resistance regime for Fn <= 0.4 is implemented, which covers
merchant ships at service speed. Higher Froude numbers raise ValueError.
"""

import math
from dataclasses import dataclass

from shiprouting.fuel.hull import Hull

G = 9.81  # m/s²
RHO_SEA = 1025.0  # kg/m³
NU_SEA = 1.18831e-6  # m²/s, sea water at 15 °C (ITTC)


@dataclass(frozen=True)
class Resistance:
    """Resistance components in newtons."""

    R_F: float  # frictional (ITTC-1957), without form factor
    form_factor: float  # 1 + k1
    R_APP: float
    R_W: float
    R_B: float
    R_TR: float
    R_A: float

    @property
    def total(self) -> float:
        return (
            self.R_F * self.form_factor
            + self.R_APP
            + self.R_W
            + self.R_B
            + self.R_TR
            + self.R_A
        )


def wetted_surface(hull: Hull) -> float:
    """Wetted surface of the bare hull, m²."""
    L, B, T = hull.L, hull.B, hull.T
    return L * (2 * T + B) * math.sqrt(hull.C_M) * (
        0.453
        + 0.4425 * hull.C_B
        - 0.2862 * hull.C_M
        - 0.003467 * B / T
        + 0.3696 * hull.C_WP
    ) + 2.38 * hull.A_BT / hull.C_B


def friction_coefficient(speed: float, L: float, nu: float = NU_SEA) -> float:
    """ITTC-1957 model-ship correlation line."""
    Rn = speed * L / nu
    return 0.075 / (math.log10(Rn) - 2) ** 2


def run_length(hull: Hull) -> float:
    """Length of the run L_R, m."""
    C_P = hull.C_P
    return hull.L * (1 - C_P + 0.06 * C_P * hull.lcb / (4 * C_P - 1))


def form_factor(hull: Hull) -> float:
    """1 + k1."""
    C_P = hull.C_P
    T_L = hull.T / hull.L
    if T_L > 0.05:
        c12 = T_L**0.2228446
    elif T_L > 0.02:
        c12 = 48.20 * (T_L - 0.02) ** 2.078 + 0.479948
    else:
        c12 = 0.479948
    c13 = 1 + 0.003 * hull.C_stern
    return c13 * (
        0.93
        + c12
        * (hull.B / run_length(hull)) ** 0.92497
        * (0.95 - C_P) ** -0.521448
        * (1 - C_P + 0.0225 * hull.lcb) ** 0.6906
    )


def half_entrance_angle(hull: Hull) -> float:
    """Half angle of entrance i_E, degrees."""
    L, B, C_P = hull.L, hull.B, hull.C_P
    return 1 + 89 * math.exp(
        -((L / B) ** 0.80856)
        * (1 - hull.C_WP) ** 0.30484
        * (1 - C_P - 0.0225 * hull.lcb) ** 0.6367
        * (run_length(hull) / B) ** 0.34574
        * (100 * hull.volume / L**3) ** 0.16302
    )


def _bulb_c2(hull: Hull) -> float:
    c3 = 0.56 * hull.A_BT**1.5 / (
        hull.B * hull.T * (0.31 * math.sqrt(hull.A_BT) + hull.T_F - hull.h_B)
    )
    return math.exp(-1.89 * math.sqrt(c3))


def wave_resistance(hull: Hull, speed: float, rho: float = RHO_SEA) -> float:
    """Wave-making and wave-breaking resistance R_W, N (Fn <= 0.4)."""
    L, B, T, C_P, volume = hull.L, hull.B, hull.T, hull.C_P, hull.volume
    Fn = speed / math.sqrt(G * L)
    if Fn > 0.4:
        raise ValueError(f"Fn={Fn:.3f} is above the implemented range (<= 0.4)")

    if B / L < 0.11:
        c7 = 0.229577 * (B / L) ** 0.33333
    elif B / L < 0.25:
        c7 = B / L
    else:
        c7 = 0.5 - 0.0625 * L / B
    c1 = (
        2223105
        * c7**3.78613
        * (T / B) ** 1.07961
        * (90 - half_entrance_angle(hull)) ** -1.37565
    )
    c2 = _bulb_c2(hull)
    c5 = 1 - 0.8 * hull.A_T / (B * T * hull.C_M)

    if C_P < 0.8:
        c16 = 8.07981 * C_P - 13.8673 * C_P**2 + 6.984388 * C_P**3
    else:
        c16 = 1.73014 - 0.7067 * C_P
    m1 = 0.0140407 * L / T - 1.75254 * volume ** (1 / 3) / L - 4.79323 * B / L - c16

    slenderness = L**3 / volume
    if slenderness < 512:
        c15 = -1.69385
    elif slenderness < 1727:
        c15 = -1.69385 + (L / volume ** (1 / 3) - 8) / 2.36
    else:
        c15 = 0.0
    m2 = c15 * C_P**2 * math.exp(-0.1 * Fn**-2)

    lam = 1.446 * C_P - 0.03 * L / B if L / B < 12 else 1.446 * C_P - 0.36
    d = -0.9
    return (
        c1 * c2 * c5 * volume * rho * G
        * math.exp(m1 * Fn**d + m2 * math.cos(lam * Fn**-2))
    )


def bulb_resistance(hull: Hull, speed: float, rho: float = RHO_SEA) -> float:
    """Additional pressure resistance of a bulbous bow near the surface R_B, N."""
    A_BT = hull.A_BT
    if A_BT == 0:
        return 0.0
    P_B = 0.56 * math.sqrt(A_BT) / (hull.T_F - 1.5 * hull.h_B)
    Fni = speed / math.sqrt(
        G * (hull.T_F - hull.h_B - 0.25 * math.sqrt(A_BT)) + 0.15 * speed**2
    )
    return 0.11 * math.exp(-3 * P_B**-2) * Fni**3 * A_BT**1.5 * rho * G / (1 + Fni**2)


def transom_resistance(hull: Hull, speed: float, rho: float = RHO_SEA) -> float:
    """Additional pressure resistance of an immersed transom R_TR, N."""
    if hull.A_T == 0:
        return 0.0
    Fn_T = speed / math.sqrt(2 * G * hull.A_T / (hull.B + hull.B * hull.C_WP))
    c6 = 0.2 * (1 - 0.2 * Fn_T) if Fn_T < 5 else 0.0
    return 0.5 * rho * speed**2 * hull.A_T * c6


def correlation_allowance(hull: Hull) -> float:
    """Model-ship correlation allowance C_A."""
    c4 = min(hull.T_F / hull.L, 0.04)
    return (
        0.006 * (hull.L + 100) ** -0.16
        - 0.00205
        + 0.003 * math.sqrt(hull.L / 7.5) * hull.C_B**4 * _bulb_c2(hull) * (0.04 - c4)
    )


def calm_water_resistance(
    hull: Hull, speed: float, rho: float = RHO_SEA, nu: float = NU_SEA
) -> Resistance:
    """Calm-water resistance components at ``speed`` (m/s through water, > 0)."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    S = wetted_surface(hull)
    C_F = friction_coefficient(speed, hull.L, nu)
    q = 0.5 * rho * speed**2
    return Resistance(
        R_F=q * S * C_F,
        form_factor=form_factor(hull),
        R_APP=q * hull.S_APP * hull.k2 * C_F,
        R_W=wave_resistance(hull, speed, rho),
        R_B=bulb_resistance(hull, speed, rho),
        R_TR=transom_resistance(hull, speed, rho),
        R_A=q * S * correlation_allowance(hull),
    )
