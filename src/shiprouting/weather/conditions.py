"""Forecast values at a point, turned into the fuel model's sea state."""

import math
from dataclasses import dataclass

from shiprouting.fuel.ship import SeaState, WaveSystem
from shiprouting.weather.forecast import Forecast

# The ITTC spectrum S = A/ω⁵ exp(-B/ω⁴) has spectral moments
# m_n = A/4 · B^((n-4)/4) · Γ((4-n)/4), so T01 / T_m-1,0 = 1 / (Γ(3/4) Γ(5/4)).
T01_PER_TM10 = 1 / (math.gamma(0.75) * math.gamma(1.25))  # ≈ 0.900


@dataclass(frozen=True)
class Weather:
    wind_u: float  # eastward wind at 10 m, m/s
    wind_v: float  # northward wind at 10 m, m/s
    Hs: float  # significant wave height, m (NaN: no wave data, e.g. land or ice)
    wave_from: float  # direction waves come from, rad clockwise from north
    T_m10: float  # mean wave period from the m-1 moment, s
    current_u: float  # eastward surface current, m/s
    current_v: float  # northward surface current, m/s


def weather_at(forecast: Forecast, lat: float, lon: float, time) -> Weather:
    def get(name: str) -> float:
        return float(forecast.sample(name, lat, lon, time))

    return Weather(
        wind_u=get("wind_u"),
        wind_v=get("wind_v"),
        Hs=get("Hs"),
        wave_from=math.atan2(get("wave_from_x"), get("wave_from_y")) % (2 * math.pi),
        T_m10=get("T_m10"),
        current_u=get("current_u"),
        current_v=get("current_v"),
    )


def sea_state(weather: Weather, heading: float) -> SeaState:
    """Sea state seen from a ship on ``heading`` (rad clockwise from north).

    Wind sea and swell come combined in one wave system; it is given the wind
    sea spreading (s = 1) because the forecast does not split them.
    """
    if math.isnan(weather.Hs):
        raise ValueError("no wave data at this point")
    wind_from = math.atan2(-weather.wind_u, -weather.wind_v)
    waves = ()
    if weather.Hs > 0:
        waves = (WaveSystem(Hs=weather.Hs, T01=T01_PER_TM10 * weather.T_m10, angle=weather.wave_from - heading),)
    return SeaState(
        wind_speed=math.hypot(weather.wind_u, weather.wind_v),
        wind_angle=wind_from - heading,
        waves=waves,
    )
