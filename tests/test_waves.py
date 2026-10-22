import dataclasses
import math

import pytest

from shiprouting.fuel import waves
from shiprouting.fuel.holtrop import G
from shiprouting.fuel.ships import KCS_SEAKEEPING

KNOT = 1852 / 3600


def test_snnm_head_waves_against_kcs_model_test():
    # Tokyo 2015 case 2.10, KCS Model 2 (L_PP 6.0702 m), U 2.017 m/s, head waves.
    # σ_aw = R_AW / (ρ g ζ_A² B² / L); measured values from the workshop data.
    scale = KCS_SEAKEEPING.L / 6.0702
    model = dataclasses.replace(
        KCS_SEAKEEPING,
        L=KCS_SEAKEEPING.L / scale,
        B=KCS_SEAKEEPING.B / scale,
        T_F=KCS_SEAKEEPING.T_F / scale,
        T_A=KCS_SEAKEEPING.T_A / scale,
    )
    measured = {0.651: 3.54, 0.851: 6.04, 1.150: 10.01, 1.371: 6.61, 1.951: 1.93}
    rho = 1000.0
    norm = rho * G * model.B**2 / model.L
    sigma = {}
    for ratio in measured:
        omega = math.sqrt(2 * math.pi * G / (ratio * model.L))
        sigma[ratio] = waves.regular_wave_resistance(model, omega, 0.0, 2.017, rho) / norm
    # Peak (λ/L ≈ 1.15) within 5 %, the rest within 2 σ_aw (≈ 20 % of the peak).
    assert sigma[1.150] == pytest.approx(measured[1.150], rel=0.05)
    for ratio, value in measured.items():
        assert sigma[ratio] == pytest.approx(value, abs=2.0)


def test_spectrum_variance_is_hs_squared_over_16():
    Hs, T01 = 3.0, 8.0
    dw = 1e-3
    m0 = sum(waves.spectrum((i + 0.5) * dw, Hs, T01) for i in range(1, 10000)) * dw
    assert m0 == pytest.approx(Hs**2 / 16, rel=3e-3)


@pytest.mark.parametrize("s", [1.0, 75.0])
def test_spreading_integrates_to_one(s):
    n = 2000
    da = math.pi / n
    total = sum(waves.spreading(-math.pi / 2 + (j + 0.5) * da, s) for j in range(n)) * da
    assert total == pytest.approx(1.0, rel=1e-6)


def test_added_resistance_scales_with_hs_squared_and_is_symmetric():
    V = 20 * KNOT
    R1 = waves.added_resistance(KCS_SEAKEEPING, 1.5, 7.0, 0.5, V)
    R2 = waves.added_resistance(KCS_SEAKEEPING, 3.0, 7.0, 0.5, V)
    assert R2 == pytest.approx(4 * R1)
    assert waves.added_resistance(KCS_SEAKEEPING, 3.0, 7.0, -0.5, V) == pytest.approx(R2)
    assert waves.added_resistance(KCS_SEAKEEPING, 0.0, 7.0, 0.0, V) == 0.0


def test_following_seas_cost_less_than_head_seas():
    V = 20 * KNOT
    head = waves.added_resistance(KCS_SEAKEEPING, 3.0, 8.0, 0.0, V)
    following = waves.added_resistance(KCS_SEAKEEPING, 3.0, 8.0, math.pi, V)
    assert 0 < following < head


def test_stern_oblique_motion_part_interpolates_beam_and_following_values():
    # ITTC: R_AWM for π/2 ≤ α ≤ π is linear between the values in beam and following waves,
    # each evaluated entirely at its own angle (including ω̄).
    V = 16 * KNOT
    for ratio in (0.5, 0.8, 1.0, 1.2):
        omega = math.sqrt(2 * math.pi * G / (ratio * KCS_SEAKEEPING.L))
        beam = waves._mean_wave_resistance(KCS_SEAKEEPING, omega, math.pi / 2, V, 1025.0)
        following = waves._mean_wave_resistance(KCS_SEAKEEPING, omega, math.pi, V, 1025.0)
        for deg in (100, 135, 170):
            w = (deg - 90) / 90
            R = waves._mean_wave_resistance(KCS_SEAKEEPING, omega, math.radians(deg), V, 1025.0)
            assert R == pytest.approx((1 - w) * beam + w * following)


def test_table_matches_the_direct_integral():
    speed = 16 * KNOT
    table = waves.AddedResistanceTable.build(KCS_SEAKEEPING, speed, T01=(6.0, 7.0, 8.0), angle=(0.0, 0.1, 0.2, math.pi))
    # On a grid point it is the integral itself, scaled by Hs².
    assert table(3.0, 7.0, -0.1) == pytest.approx(waves.added_resistance(KCS_SEAKEEPING, 3.0, 7.0, 0.1, speed))
    # Between grid points it interpolates, close to the integral.
    assert table(2.0, 6.6, 0.15) == pytest.approx(waves.added_resistance(KCS_SEAKEEPING, 2.0, 6.6, 0.15, speed), rel=0.02)
    assert table(0.0, 7.0, 0.0) == 0.0
