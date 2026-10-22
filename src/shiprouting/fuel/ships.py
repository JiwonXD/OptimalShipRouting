"""Reference ships."""

from shiprouting.fuel.hull import Hull

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
