"""Main engine fuel consumption."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Engine:
    """Two-stroke main engine.

    SFOC is the parabola through three published (load, SFOC) points, e.g. the
    50/75/100 % values of an engine maker's programme. Below the lowest point
    the parabola is extrapolated, which is less reliable.
    """

    mcr: float  # maximum continuous rating, W
    sfoc_points: tuple[tuple[float, float], ...]  # (load fraction, g/kWh) x 3

    def sfoc(self, power: float) -> float:
        """Specific fuel oil consumption at brake ``power`` (W), g/kWh."""
        load = power / self.mcr
        if not 0 < load <= 1:
            raise ValueError(f"engine load {load:.2f} is outside (0, 1]")
        (x0, y0), (x1, y1), (x2, y2) = self.sfoc_points
        return (
            y0 * (load - x1) * (load - x2) / ((x0 - x1) * (x0 - x2))
            + y1 * (load - x0) * (load - x2) / ((x1 - x0) * (x1 - x2))
            + y2 * (load - x0) * (load - x1) / ((x2 - x0) * (x2 - x1))
        )

    def fuel_rate(self, power: float) -> float:
        """Fuel consumption at brake ``power`` (W), kg/h."""
        return power / 1e3 * self.sfoc(power) / 1e3
