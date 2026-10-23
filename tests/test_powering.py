import math

import pytest

from shiprouting.fuel import powering
from shiprouting.fuel.ships import KCS_OPEN_WATER, KCS_PROPELLER_D

CURVE, D = KCS_OPEN_WATER, KCS_PROPELLER_D


def test_brake_power_matches_propulsive_efficiency_chain():
    # P_B = R V / (η_H η_O η_R η_S) with η_H = (1 - t) / (1 - w).
    factors = powering.PropulsionFactors(t=0.15, w=0.2, eta_R=1.01, eta_S=0.99)
    R, V, rho = 1.6e6, 12.35, 1025.0
    P_B = powering.brake_power(R, V, factors, CURVE, D, rho)

    thrust, V_A = R / 0.85, V * 0.8
    c = thrust / (rho * D**2 * V_A**2)
    J = next(j / 1e5 for j in range(50000, 100000) if CURVE.k_t(j / 1e5) <= c * (j / 1e5) ** 2)
    eta_O = J * CURVE.k_t(J) / (2 * math.pi * CURVE.k_q(J))
    expected = R * V / ((0.85 / 0.8) * eta_O * 1.01 * 0.99)
    assert P_B == pytest.approx(expected, rel=1e-4)
