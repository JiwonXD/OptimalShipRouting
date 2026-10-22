"""Reference ships."""

import math

from shiprouting.fuel import model_test
from shiprouting.fuel.engine import Engine
from shiprouting.fuel.hull import Hull
from shiprouting.fuel.model_test import ResistanceTest
from shiprouting.fuel.powering import PropulsionFactors
from shiprouting.fuel.propeller import OpenWaterCurve
from shiprouting.fuel.ship import Ship
from shiprouting.fuel.waves import Seakeeping
from shiprouting.fuel.wind import Windage

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

# KP505 propeller open-water test (model D 0.25 m), Tokyo 2015 openwater.txt.
# Full scale: D 7.9 m, 5 blades, P/D(0.7R) 0.997, A_E/A_0 0.800.
KCS_PROPELLER_D = 7.9
KCS_OPEN_WATER = OpenWaterCurve(
    J=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05),
    K_T=(0.518, 0.482, 0.435, 0.387, 0.336, 0.285, 0.235, 0.185, 0.161, 0.137,
         0.111, 0.083, 0.053, 0.022, -0.012),
    K_Q=(0.0725, 0.0677, 0.0622, 0.0557, 0.0497, 0.0437, 0.0376, 0.0311, 0.0278,
         0.0247, 0.0215, 0.0181, 0.0140, 0.0096, 0.0046),
)

# Propulsion factors at model scale (full-scale wake correction not applied).
# w and η_R: thrust identity on the case 2.5 self-propulsion point (K_T 0.170,
# K_Q 0.0288, n 9.5 rps, U 2.196 m/s) with the KP505 curve.
# t: not in the published data; Holtrop (1984) regression gives 0.180.
# η_S: assumed.
KCS_PROPULSION = PropulsionFactors(t=0.180, w=0.209, eta_R=1.008, eta_S=0.99)

# KCS was never built, so the engine is assumed: MAN G80ME-C10.5, Tier II,
# high-load tuning (4,710 kW per cylinder, Marine Engine Programme 2021).
# Sizing rule (assumed): power at 24 kn in calm water and still air (≈29.6 MW)
# plus 15 % sea margin at about 90 % MCR, so 8 cylinders = 37,680 kW.
# SFOC at 50/75/100 % load, ISO conditions, LCV 42,700 kJ/kg.
KCS_ENGINE = Engine(
    mcr=8 * 4710e3,
    sfoc_points=((0.50, 160.5), (0.75, 159.5), (1.00, 164.0)),
)

# Wind: ITTC 7.5-04-01-01.1 (Rev 08) Table F-1, container ship (6800 TEU),
# laden with containers, C_DA = -C_X. KCS has no published superstructure, so
# A_XV is assumed: B 32.2 m x about 25 m (freeboard 8.2 m + deck containers
# and bridge).
KCS_WINDAGE = Windage(
    A_XV=800.0,
    angles=(0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180),
    C_DA=(0.67, 0.76, 0.66, 0.40, 0.25, 0.21, 0.26, 0.04, -0.36, -0.69, -0.90,
          -0.85, -0.67),
)

# Waves (SNNM). k_yy 0.25 L_PP as in the Tokyo 2015 seakeeping tests (case 2.10).
# E1, E2 provisional: E1 from the Holtrop i_E regression (15.5°), E2 assumed
# until both are measured on the hull geometry.
KCS_SEAKEEPING = Seakeeping(
    L=KCS_L_PP,
    B=32.2,
    T_F=10.8,
    T_A=10.8,
    C_B=0.6505,
    k_yy=0.25,
    E1=math.radians(15.5),
    E2=math.radians(25.0),
)

KCS_SHIP = Ship(
    resistance=lambda speed: model_test.calm_water_resistance(
        KCS_RESISTANCE_TEST, KCS_L_PP, speed
    ),
    factors=KCS_PROPULSION,
    propeller=KCS_OPEN_WATER,
    propeller_D=KCS_PROPELLER_D,
    engine=KCS_ENGINE,
    windage=KCS_WINDAGE,
    seakeeping=KCS_SEAKEEPING,
)
