import numpy as np
import pytest

from shiprouting.weather.forecast import Forecast

T0 = np.datetime64("2026-10-21T18:00", "s")
LAT = np.arange(-90.0, 90.1, 1.0)
LON = np.arange(-180.0, 180.0, 1.0)
TIME = np.array([T0, T0 + np.timedelta64(24, "h")])


def make(values):
    return Forecast(lat=LAT, lon=LON, time=TIME, fields={"f": values})


def test_bilinear_in_space_and_linear_in_time_is_exact_for_linear_fields():
    lat, lon = np.meshgrid(LAT, LON, indexing="ij")
    values = np.stack([2 * lat + 0.5 * lon, 2 * lat + 0.5 * lon + 10])
    fc = make(values)
    t = T0 + np.timedelta64(6, "h")
    assert fc.sample("f", 35.3, 140.7, t) == pytest.approx(2 * 35.3 + 0.5 * 140.7 + 2.5)
    # Arrays broadcast.
    out = fc.sample("f", [10.25, -20.5], 30.0, T0)
    assert out == pytest.approx([2 * 10.25 + 15, -41 + 15])


def test_longitude_wraps_across_the_dateline():
    values = np.zeros((2, len(LAT), len(LON)))
    values[:, :, LON == 179.0] = 1.0
    values[:, :, LON == -180.0] = 3.0
    fc = make(values)
    assert fc.sample("f", 0.0, 179.5, T0) == pytest.approx(2.0)
    assert fc.sample("f", 0.0, -180.5 + 360, T0) == pytest.approx(2.0)
    assert fc.sample("f", 0.0, 180.0, T0) == pytest.approx(3.0)


def test_missing_points_are_left_out():
    values = np.ones((2, len(LAT), len(LON)))
    land = (LON >= 0)[None, None, :] & np.ones((2, len(LAT), 1), bool)
    values[land] = np.nan
    fc = make(values)
    assert fc.sample("f", 0.0, -0.5, T0) == pytest.approx(1.0)  # next to the coast
    assert np.isnan(fc.sample("f", 0.0, 10.0, T0))  # inland


def test_time_outside_forecast_is_rejected():
    fc = make(np.zeros((2, len(LAT), len(LON))))
    with pytest.raises(ValueError):
        fc.sample("f", 0.0, 0.0, T0 + np.timedelta64(25, "h"))
