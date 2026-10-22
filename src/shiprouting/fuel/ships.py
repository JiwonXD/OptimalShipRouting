"""Reference ships."""

from shiprouting.fuel.hull import Hull
from shiprouting.fuel.model_test import ResistanceTest

# KRISO Container Ship (KCS), full scale.
# Published (SIMMAN 2008, Tokyo 2015 CFD Workshop): L_WL 232.5 m, L_PP 230 m,
# B 32.2 m, T 10.8 m, volume 52030 m³, C_M 0.9849, LCB -1.48 % L_PP,
# rudder wetted area 115 m², bare-hull wetted surface 9424 m² (T2015).
# Not found in the sources read, so these are provisional until derived from
# the hull geometry: C_WP (set so the Holtrop wetted surface is within 0.5 %
# of 9424 m²), A_BT, h_B, A_T (transom assumed dry at rest), C_stern, k2.
KCS = Hull(
    L=232.5,
    B=32.2,
    T_F=10.8,
    T_A=10.8,
    volume=52030.0,
    lcb=-1.48 * 230.0 / 232.5,
    C_M=0.9849,
    C_WP=0.82,
    A_BT=27.0,
    h_B=4.3,
    A_T=0.0,
    S_APP=115.0,
    k2=1.5,
    C_stern=0.0,
)

# KCS Model 1 resistance test, Tokyo 2015 CFD Workshop case 2.1
# (L_PP 7.2786 m, with rudder, free sinkage and trim, S0/L_PP² = 0.1803,
# ν = 1.27e-6 m²/s). Measurement uncertainty of C_T is 1.0 %.
KCS_RESISTANCE_TEST = ResistanceTest(
    L=7.2786,
    wetted_surface_ratio=0.1803,
    nu=1.27e-6,
    froude=(0.108, 0.152, 0.195, 0.227, 0.260, 0.282),
    C_T=(3.796e-3, 3.641e-3, 3.475e-3, 3.467e-3, 3.711e-3, 4.501e-3),
)
KCS_L_PP = 230.0
