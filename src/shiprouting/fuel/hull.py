"""Hull particulars used by the resistance and propulsion models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hull:
    """Hull inputs for the Holtrop-Mennen (1982) method.

    Units are SI (m, m², m³). ``lcb`` is the longitudinal centre of buoyancy
    in percent of L, forward of 0.5L positive.
    """

    L: float  # waterline length
    B: float  # moulded breadth
    T_F: float  # draught at forward perpendicular
    T_A: float  # draught at aft perpendicular
    volume: float  # moulded displacement volume ∇
    lcb: float  # % of L, forward of 0.5L positive
    C_M: float  # midship section coefficient
    C_WP: float  # waterplane area coefficient
    A_BT: float  # transverse bulb area at the forward perpendicular
    h_B: float  # height of the bulb centroid above the keel
    A_T: float  # immersed transom area at rest
    S_APP: float  # wetted area of appendages
    k2: float  # appendage form factor (1 + k2)
    C_stern: float  # stern shape: -25 pram/gondola, -10 V, 0 normal, +10 U (Hogner)

    @property
    def T(self) -> float:
        return 0.5 * (self.T_F + self.T_A)

    @property
    def C_B(self) -> float:
        return self.volume / (self.L * self.B * self.T)

    @property
    def C_P(self) -> float:
        return self.C_B / self.C_M
