"""Added resistance in waves, ITTC 7.5-04-01-01.1 (Rev 08, 2024), Annex G.

Regular-wave transfer function: SNNM (Liu & Papanikolaou), valid for all wave
headings. Irregular waves: modified Pierson-Moskowitz spectrum with cos^2s
directional spreading,

    R_AW = 2 ∫∫ R_wave(ω, α) / ζ_A² · S(ω) · G(α) dω dα

Wave angles are where the waves come from relative to the bow: 0 = head seas,
π = following seas, either side.
"""

import math
from dataclasses import dataclass

from shiprouting.fuel.holtrop import G, RHO_SEA


@dataclass(frozen=True)
class Seakeeping:
    """Hull data the SNNM formula needs."""

    L: float  # length between perpendiculars, m
    B: float  # m
    T_F: float  # draught at forward perpendicular, m
    T_A: float  # draught at aft perpendicular, m
    C_B: float  # block coefficient
    k_yy: float  # pitch radius of gyration / L
    E1: float  # half entrance angle of the waterline, rad
    E2: float  # half run angle of the waterline, rad


def _mean_wave_resistance(sk: Seakeeping, omega: float, alpha: float, speed: float, rho: float) -> float:
    """Motion-induced part R_AWM / ζ_A² at heading ``alpha`` (0..π)."""
    L, B, C_B = sk.L, sk.B, sk.C_B
    T = max(sk.T_F, sk.T_A)
    trim = math.atan(abs(sk.T_A - sk.T_F) / L)
    Fr = speed / math.sqrt(G * L)
    V_g = G / (2 * omega)  # group velocity of deep-water waves
    Fr_rel = (speed - V_g) / math.sqrt(G * L)

    omega_bar = (
        2.142
        * sk.k_yy ** (1 / 3)
        * math.sqrt(L / (2 * math.pi * G))
        * (C_B / 0.65) ** 0.17
        * (1 - 0.111 / C_B * (math.log(B / T) - math.log(2.75)))
        * ((-1.377 * Fr**2 + 1.157 * Fr) * abs(math.cos(alpha)) + 0.618 * (13 + math.cos(2 * alpha)) / 14)
        * omega
    )
    if omega_bar < 1:
        b1, d1 = 11.0, 566 * (L * C_B / B) ** -2.66
    else:
        b1, d1 = -8.5, -566 * (L / B) ** -2.66 * (4 - 125 * trim)
    a3 = 1.0 + 28.7 * trim

    def at(a: float) -> float:
        if a <= math.pi / 2:
            a1 = (0.87 / C_B) ** ((1 + Fr) * math.cos(a)) / math.log(B / T) * (1 + 2 * math.cos(a)) / 3
            a2 = 0.0072 + 0.1676 * Fr if Fr < 0.12 else Fr**1.5 * math.exp(-3.5 * Fr)
        else:
            if speed > V_g and Fr_rel >= 0.12:
                a1 = (0.87 / C_B) ** (1 + Fr_rel) / math.log(B / T)
            else:
                a1 = (0.87 / C_B) / math.log(B / T)
            if speed <= V_g:
                a2 = 0.0072 * (2 * speed / V_g - 1)
            elif Fr_rel < 0.12:
                a2 = 0.0072 + 0.1676 * Fr_rel
            else:
                a2 = Fr_rel**1.5 * math.exp(-3.5 * Fr_rel)
        return (
            3859.2 * rho * G * B**2 / L * C_B**1.34 * sk.k_yy**2
            * a1 * a2 * a3
            * omega_bar**b1 * math.exp(b1 / d1 * (1 - omega_bar**d1))
        )

    if alpha <= math.pi / 2:
        return at(alpha)
    # Stern-oblique seas: linear between beam and following seas.
    w = (alpha - math.pi / 2) / (math.pi / 2)
    return (1 - w) * at(math.pi / 2) + w * at(math.pi)


def _reflection_resistance(sk: Seakeeping, omega: float, alpha: float, speed: float, rho: float) -> float:
    """Wave-reflection part R_AWR / ζ_A² at heading ``alpha`` (0..π)."""
    L, B, C_B, E1, E2 = sk.L, sk.B, sk.C_B, sk.E1, sk.E2
    T = max(sk.T_F, sk.T_A)
    Fr = speed / math.sqrt(G * L)
    wavelength = 2 * math.pi * G / omega**2
    if wavelength / L > 2.5:
        return 0.0

    def alpha_T(T_star: float) -> float:
        return 1 - math.exp(-4 * math.pi * (T_star / wavelength - T_star / (2.5 * L)))

    cos_a = math.cos(alpha)
    k = 2 * omega * speed / G
    c = 2.25 / 4 * rho * G * B
    bow = (0.87 / C_B) ** ((1 + 4 * math.sqrt(Fr)) * (cos_a if alpha <= E1 else 0.0))
    if C_B <= 0.75:
        T_stern = T * (4 + math.sqrt(abs(cos_a))) / 5
    else:
        T_stern = T * (2 + math.sqrt(abs(cos_a))) / 3

    R = 0.0
    if alpha <= math.pi - E1:
        R += c * alpha_T(T) * (math.sin(E1 + alpha) ** 2 + k * (cos_a - math.cos(E1) * math.cos(E1 + alpha))) * bow
    if alpha <= E1:
        R += c * alpha_T(T) * (math.sin(E1 - alpha) ** 2 + k * (cos_a - math.cos(E1) * math.cos(E1 - alpha))) * bow
    if alpha >= E2:
        R -= c * alpha_T(T_stern) * (math.sin(E2 - alpha) ** 2 + k * (cos_a - math.cos(E2) * math.cos(E2 - alpha)))
    if alpha >= math.pi - E2:
        R -= c * alpha_T(T_stern) * (math.sin(E2 + alpha) ** 2 + k * (cos_a - math.cos(E2) * math.cos(E2 + alpha)))
    return R


def regular_wave_resistance(
    sk: Seakeeping, omega: float, alpha: float, speed: float, rho: float = RHO_SEA
) -> float:
    """Added resistance per squared wave amplitude, R_wave / ζ_A², N/m².

    ``omega`` is the wave frequency (rad/s), ``alpha`` the wave angle (rad) and
    ``speed`` the speed through water (m/s).
    """
    a = abs(math.remainder(alpha, 2 * math.pi))
    return _mean_wave_resistance(sk, omega, a, speed, rho) + _reflection_resistance(sk, omega, a, speed, rho)


def spectrum(omega: float, Hs: float, T01: float) -> float:
    """Modified Pierson-Moskowitz spectrum, m²·s, with mean period T01 (s)."""
    A = 173 * Hs**2 / T01**4
    B = 691 / T01**4
    return A / omega**5 * math.exp(-B / omega**4)


def spreading(offset: float, s: float) -> float:
    """cos^2s spreading for a direction ``offset`` (rad) from the mean, |offset| ≤ π/2.

    The ITTC normalisation 2^2s/π · Γ²(s+1)/Γ(2s+1) is written as
    Γ(s+1) / (√π Γ(s+½)) so that large s (swell, s = 75) does not overflow.
    """
    norm = math.exp(math.lgamma(s + 1) - math.lgamma(s + 0.5)) / math.sqrt(math.pi)
    return norm * math.cos(offset) ** (2 * s)


def added_resistance(
    sk: Seakeeping,
    Hs: float,
    T01: float,
    wave_angle: float,
    speed: float,
    s: float = 1.0,
    rho: float = RHO_SEA,
    n_freq: int = 80,
    n_dir: int = 91,
) -> float:
    """Mean added resistance in irregular short-crested waves, N.

    ``Hs`` is the significant wave height (m), ``T01`` the mean period (s),
    ``wave_angle`` the mean wave direction from the bow (rad, 0 = head) and
    ``s`` the spreading parameter (ITTC: 1 for wind sea, 75 for swell).
    """
    if Hs <= 0:
        return 0.0
    omega_p = (4 / 5 * 691 / T01**4) ** 0.25  # spectral peak
    w0, w1 = 0.2 * omega_p, 5.0 * omega_p
    dw = (w1 - w0) / n_freq
    omegas = [w0 + (i + 0.5) * dw for i in range(n_freq)]
    da = math.pi / n_dir
    offsets = [-math.pi / 2 + (j + 0.5) * da for j in range(n_dir)]
    weights = [spreading(o, s) for o in offsets]
    total_weight = sum(weights) * da  # renormalise away the quadrature error

    R = 0.0
    for o, wgt in zip(offsets, weights):
        alpha = wave_angle + o
        inner = sum(regular_wave_resistance(sk, w, alpha, speed, rho) * spectrum(w, Hs, T01) for w in omegas)
        R += wgt * inner
    return 2 * R * dw * da / total_weight
