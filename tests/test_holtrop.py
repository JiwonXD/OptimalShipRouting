import pytest

from shiprouting.fuel import holtrop
from shiprouting.fuel.hull import Hull
from shiprouting.fuel.ships import KCS

KNOT = 1852 / 3600

# Numerical example in Holtrop & Mennen (1982).
EXAMPLE_HULL = Hull(
    L=205.0,
    B=32.0,
    T_F=10.0,
    T_A=10.0,
    volume=37500.0,
    lcb=-0.75,
    C_M=0.98,
    C_WP=0.75,
    A_BT=20.0,
    h_B=4.0,
    A_T=16.0,
    S_APP=50.0,
    k2=1.5,
    C_stern=10.0,
)
EXAMPLE_SPEED = 25 * KNOT


@pytest.fixture
def example():
    return holtrop.calm_water_resistance(EXAMPLE_HULL, EXAMPLE_SPEED)


def test_example_total_resistance_and_effective_power(example):
    # Paper: R_T = 1793.26 kN, P_E = 23063 kW. The paper rounds intermediate
    # values, so allow 0.2 %.
    assert example.total / 1e3 == pytest.approx(1793.26, rel=2e-3)
    assert example.total * EXAMPLE_SPEED / 1e3 == pytest.approx(23063, rel=2e-3)


def test_example_components(example):
    # Intermediate values printed in the worked example.
    assert holtrop.wetted_surface(EXAMPLE_HULL) == pytest.approx(7381.45, abs=0.5)
    assert holtrop.half_entrance_angle(EXAMPLE_HULL) == pytest.approx(12.08, abs=0.01)
    assert example.form_factor == pytest.approx(1.156, abs=1e-3)
    assert example.R_F / 1e3 == pytest.approx(869.63, rel=1e-3)
    assert example.R_APP / 1e3 == pytest.approx(8.83, rel=5e-3)
    assert example.R_W / 1e3 == pytest.approx(557.11, rel=2e-3)
    assert example.R_B / 1e3 < 0.1
    assert example.R_TR == 0.0


def test_kcs_wetted_surface_matches_published():
    assert holtrop.wetted_surface(KCS) == pytest.approx(9424, rel=5e-3)


def test_kcs_resistance_increases_with_speed():
    totals = [
        holtrop.calm_water_resistance(KCS, v * KNOT).total for v in range(8, 27, 2)
    ]
    assert totals == sorted(totals)


def test_rejects_speed_outside_range():
    with pytest.raises(ValueError):
        holtrop.calm_water_resistance(KCS, 0.0)
    with pytest.raises(ValueError):
        holtrop.calm_water_resistance(KCS, 40 * KNOT)  # Fn ≈ 0.43
