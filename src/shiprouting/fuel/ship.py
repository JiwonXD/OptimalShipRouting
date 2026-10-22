"""A ship's calm-water fuel model."""

from collections.abc import Callable
from dataclasses import dataclass

from shiprouting.fuel.engine import Engine
from shiprouting.fuel.powering import PropulsionFactors, brake_power
from shiprouting.fuel.propeller import OpenWaterCurve


@dataclass(frozen=True)
class Ship:
    resistance: Callable[[float], float]  # calm-water resistance (N) at speed (m/s)
    factors: PropulsionFactors
    propeller: OpenWaterCurve
    propeller_D: float  # m
    engine: Engine

    def brake_power(self, speed: float) -> float:
        """Brake power in calm water, W, at ``speed`` (m/s through water)."""
        return brake_power(
            self.resistance(speed), speed, self.factors, self.propeller, self.propeller_D
        )

    def fuel_rate(self, speed: float) -> float:
        """Fuel consumption in calm water, kg/h, at ``speed`` (m/s through water)."""
        return self.engine.fuel_rate(self.brake_power(speed))
