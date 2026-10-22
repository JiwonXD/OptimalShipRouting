import math

import pytest

from shiprouting.fuel import holtrop, model_test
from shiprouting.fuel.ships import KCS, KCS_L_PP, KCS_RESISTANCE_TEST

KNOT = 1852 / 3600
TEST = KCS_RESISTANCE_TEST


def model_cf(Fn):
    V = Fn * math.sqrt(9.81 * TEST.L)
    return 0.075 / (math.log10(V * TEST.L / TEST.nu) - 2) ** 2


def test_prohaska_form_factor():
    # Least-squares fit through the three points with Fn <= 0.2.
    assert model_test.prohaska_form_factor(TEST) == pytest.approx(1.133, abs=2e-3)


def test_residuary_coefficient_reproduces_measured_points():
    k = 1.13
    for Fn, C_T in zip(TEST.froude, TEST.C_T):
        C_R = model_test.residuary_coefficient(TEST, Fn, k)
        assert C_R + k * model_cf(Fn) == pytest.approx(C_T, rel=1e-12)


def test_residuary_coefficient_interpolates_linearly():
    k = 1.13
    lo, hi = TEST.froude[3], TEST.froude[4]
    mid = model_test.residuary_coefficient(TEST, 0.5 * (lo + hi), k)
    ends = [model_test.residuary_coefficient(TEST, f, k) for f in (lo, hi)]
    assert mid == pytest.approx(sum(ends) / 2, rel=1e-12)


def test_rejects_froude_outside_test_range():
    with pytest.raises(ValueError):
        model_test.residuary_coefficient(TEST, 0.30, 1.13)
    with pytest.raises(ValueError):
        model_test.residuary_coefficient(TEST, 0.10, 1.13)


def test_roughness_allowance():
    # 0.044 [(150e-6 / 230)^(1/3) - 10 Re^(-1/3)] + 0.000125 at Re = 2.39e9
    speed = 2.39e9 * holtrop.NU_SEA / 230.0
    assert model_test.roughness_allowance(speed, 230.0) == pytest.approx(1.77e-4, rel=1e-2)


def test_kcs_full_scale_at_service_speed():
    # Fn 0.26 at 230 m is a measured point, so C_R needs no interpolation.
    k = 1.133
    speed = 0.26 * math.sqrt(9.81 * KCS_L_PP)
    C_R = TEST.C_T[4] - k * model_cf(0.26)
    C_FS = 0.075 / (math.log10(speed * KCS_L_PP / holtrop.NU_SEA) - 2) ** 2
    dC_F = model_test.roughness_allowance(speed, KCS_L_PP)
    expected = 0.5 * 1025 * speed**2 * 0.1803 * KCS_L_PP**2 * (k * C_FS + C_R + dC_F)

    R = model_test.calm_water_resistance(TEST, KCS_L_PP, speed, form_factor=k)
    assert R == pytest.approx(expected, rel=1e-9)
    # About 24 kn; the test-based value is well below the Holtrop estimate.
    assert 1600e3 < R < 1700e3
    assert R < holtrop.calm_water_resistance(KCS, speed).total
