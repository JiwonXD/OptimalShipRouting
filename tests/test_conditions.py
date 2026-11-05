import math

import pytest

from shiprouting.fuel.waves import spectrum
from shiprouting.weather.conditions import T01_PER_TM10, Weather, sea_state


def test_t01_conversion_matches_the_ittc_spectrum_moments():
    Hs, T01 = 3.0, 8.0
    dw = 1e-3
    omegas = [(i + 0.5) * dw for i in range(100, 20000)]
    m_1 = sum(spectrum(w, Hs, T01) / w for w in omegas) * dw
    m0 = sum(spectrum(w, Hs, T01) for w in omegas) * dw
    m1 = sum(spectrum(w, Hs, T01) * w for w in omegas) * dw
    T_m10 = 2 * math.pi * m_1 / m0
    assert 2 * math.pi * m0 / m1 == pytest.approx(T01, rel=2e-3)
    assert T01_PER_TM10 * T_m10 == pytest.approx(T01, rel=2e-3)


def test_sea_state_angles_relative_to_heading():
    # Wind blowing toward the south (from the north), waves from the east, ship heading east.
    w = Weather(wind_u=0.0, wind_v=-10.0, Hs=2.0, wave_from=math.pi / 2, T_m10=8.0,
                current_u=0.0, current_v=0.0)
    sea = sea_state(w, heading=math.pi / 2)
    assert sea.wind_speed == pytest.approx(10.0)
    assert math.remainder(sea.wind_angle, 2 * math.pi) == pytest.approx(-math.pi / 2)  # from port
    (waves,) = sea.waves
    assert waves.angle == pytest.approx(0.0)  # head seas
    assert waves.T01 == pytest.approx(8.0 * T01_PER_TM10)


def test_no_wave_data_is_an_error():
    w = Weather(1.0, 1.0, math.nan, 0.0, math.nan, 0.0, 0.0)
    with pytest.raises(ValueError):
        sea_state(w, 0.0)
