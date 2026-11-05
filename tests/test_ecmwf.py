import math

import numpy as np
import pytest

from shiprouting.weather import ecmwf
from shiprouting.weather.conditions import weather_at

eccodes = pytest.importorskip("eccodes")


def write_field(f, param_id, step, values, bitmap=False):
    """Write a 1° global field laid out like ECMWF open data (lat 90 → -90, lon from 180)."""
    h = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(h, "paramId", param_id)
        for key, value in {
            "Ni": 360, "Nj": 181,
            "latitudeOfFirstGridPointInDegrees": 90.0, "longitudeOfFirstGridPointInDegrees": 180.0,
            "latitudeOfLastGridPointInDegrees": -90.0, "longitudeOfLastGridPointInDegrees": 179.0,
            "iDirectionIncrementInDegrees": 1.0, "jDirectionIncrementInDegrees": 1.0,
            "jScansPositively": 0,
            "dataDate": 20261021, "dataTime": 1800, "step": step,
        }.items():
            eccodes.codes_set(h, key, value)
        if bitmap:
            eccodes.codes_set(h, "bitmapPresent", 1)
            eccodes.codes_set(h, "missingValue", 9999)
        eccodes.codes_set_values(h, values.ravel())
        eccodes.codes_write(h, f)
    finally:
        eccodes.codes_release(h)


def test_load_reorders_grid_and_masks_missing(tmp_path):
    lat = 90.0 - np.arange(181)[:, None] * np.ones((1, 360))
    lon = (180.0 + np.arange(360))[None, :] % 360 * np.ones((181, 1))
    lon = (lon + 180) % 360 - 180
    sea = lon < 100  # pretend everything east of 100°E is land for wave fields
    path = tmp_path / "sample.grib2"
    with open(path, "wb") as f:
        for step in (0, 24):
            for pid, values, bm in [
                (165, lat + 0 * lon + step, False),  # 10u
                (166, -lon / 10, False),  # 10v
                (262140, np.full_like(lat, 0.5), False),  # sve
                (262139, np.zeros_like(lat), False),  # svn
                (140229, np.where(sea, 2.0, 9999), True),  # swh
                (140230, np.where(sea, 350.0 if step == 0 else 10.0, 9999), True),  # mwd
                (140232, np.where(sea, 8.0, 9999), True),  # mwp
            ]:
                write_field(f, pid, step, values, bm)

    fc = ecmwf.load([path])
    assert fc.lat[0] == -90 and fc.lon[0] == -180
    assert fc.time[1] - fc.time[0] == np.timedelta64(24, "h")
    assert sorted(fc.fields) == sorted(
        ["wind_u", "wind_v", "current_u", "current_v", "Hs", "wave_from_x", "wave_from_y", "T_m10"]
    )

    t = np.datetime64("2026-10-22T06:00", "s")  # halfway
    w = weather_at(fc, 35.0, 40.0, t)
    assert w.wind_u == pytest.approx(35.0 + 12.0, abs=1e-3)
    assert w.wind_v == pytest.approx(-4.0, abs=1e-3)
    assert w.Hs == pytest.approx(2.0)
    assert w.T_m10 == pytest.approx(8.0, abs=1e-3)
    # 350° and 10° average to north, not to 180°.
    assert math.remainder(w.wave_from, 2 * math.pi) == pytest.approx(0.0, abs=1e-3)
    assert math.isnan(weather_at(fc, 35.0, 140.0, t).Hs)
