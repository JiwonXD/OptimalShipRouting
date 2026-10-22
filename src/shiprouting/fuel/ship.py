"""A ship's fuel model in calm water, wind and waves."""

from collections.abc import Callable
from dataclasses import dataclass

from shiprouting.fuel.engine import Engine
from shiprouting.fuel.powering import PropulsionFactors, brake_power
from shiprouting.fuel.propeller import OpenWaterCurve
from shiprouting.fuel.waves import Seakeeping, added_resistance
from shiprouting.fuel.wind import Windage, air_resistance


@dataclass(frozen=True)
class WaveSystem:
    """One wave system, e.g. wind sea or a swell."""

    Hs: float  # significant wave height, m
    T01: float  # mean wave period from m0/m1, s
    angle: float  # mean direction the waves come from relative to the bow, rad (0 = head)
    s: float = 1.0  # spreading parameter (ITTC: 1 wind sea, 75 swell)


@dataclass(frozen=True)
class SeaState:
    wind_speed: float = 0.0  # true wind speed, m/s
    wind_angle: float = 0.0  # direction the wind comes from relative to the bow, rad (0 = head)
    waves: tuple[WaveSystem, ...] = ()


CALM = SeaState()


@dataclass(frozen=True)
class Ship:
    resistance: Callable[[float], float]  # calm-water resistance (N) at speed (m/s)
    factors: PropulsionFactors
    propeller: OpenWaterCurve
    propeller_D: float  # m
    engine: Engine
    windage: Windage | None = None  # None: air resistance left out
    seakeeping: Seakeeping | None = None  # None: waves not supported

    def total_resistance(self, speed: float, sea: SeaState = CALM) -> float:
        """Calm-water + air + wave added resistance, N, at ``speed`` (m/s through water).

        Without a current, speed through water is also used as speed over ground.
        """
        R = self.resistance(speed)
        if self.windage is not None:
            R += air_resistance(self.windage, sea.wind_speed, sea.wind_angle, speed)
        if sea.waves and self.seakeeping is None:
            raise ValueError("ship has no seakeeping data for waves")
        for w in sea.waves:
            R += added_resistance(self.seakeeping, w.Hs, w.T01, w.angle, speed, w.s)
        return R

    def brake_power(self, speed: float, sea: SeaState = CALM) -> float:
        """Brake power, W, at ``speed`` (m/s through water)."""
        return brake_power(
            self.total_resistance(speed, sea), speed, self.factors, self.propeller, self.propeller_D
        )

    def fuel_rate(self, speed: float, sea: SeaState = CALM) -> float:
        """Fuel consumption, kg/h, at ``speed`` (m/s through water)."""
        return self.engine.fuel_rate(self.brake_power(speed, sea))
