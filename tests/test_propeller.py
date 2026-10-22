import math

import pytest

from shiprouting.fuel import propeller
from shiprouting.fuel.ships import KCS_OPEN_WATER

CURVE = KCS_OPEN_WATER


def test_open_water_efficiency_matches_published_table():
    # η0 = J K_T / (2π K_Q); openwater.txt lists 0.665 at J 0.7 and 0.705 at
    # J 0.8. K_T and K_Q are given to three digits, hence the tolerance.
    for J, eta in ((0.7, 0.665), (0.8, 0.705)):
        assert J * CURVE.k_t(J) / (2 * math.pi * CURVE.k_q(J)) == pytest.approx(eta, abs=3e-3)


def test_operating_point_recovers_advance_ratio():
    D, rho, n = 0.25, 1000.0, 9.5
    for J in (0.3, 0.62, 0.7313, 0.9):
        V_A = J * n * D
        thrust = CURVE.k_t(J) * rho * n**2 * D**4
        point = propeller.operating_point(CURVE, thrust, V_A, D, rho)
        assert point.J == pytest.approx(J, abs=1e-9)
        assert point.n == pytest.approx(n, rel=1e-8)


def test_kcs_self_propulsion_point():
    # Tokyo 2015 case 2.5 (model, Fr 0.26): K_T 0.170, K_Q 0.0288, n 9.5 rps,
    # U 2.196 m/s. Thrust identity gives 1 - w ≈ 0.79 and η_R ≈ 1.01.
    D, rho, n, U = 0.25, 999.1, 9.5, 2.196
    J = next(
        j / 10000 for j in range(7000, 7500) if CURVE.k_t(j / 10000) <= 0.170
    )
    assert J * n * D / U == pytest.approx(0.791, abs=2e-3)
    assert CURVE.k_q(J) / 0.0288 == pytest.approx(1.008, abs=3e-3)


def test_delivered_power_is_2_pi_n_q():
    point = propeller.OperatingPoint(J=0.7, n=2.0, K_T=0.185, K_Q=0.0311)
    D, rho = 7.9, 1025.0
    torque = 0.0311 * rho * 4.0 * D**5
    assert propeller.delivered_power(point, D, rho, eta_R=1.0) == pytest.approx(
        2 * math.pi * 2.0 * torque
    )


def test_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        propeller.operating_point(CURVE, 0.0, 5.0, 7.9, 1025.0)
