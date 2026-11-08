import numpy as np
import pytest

from shiprouting.weather import era5


def write_day(path, day, every_hours=12):
    """A day of fields as ARCO-ERA5 stores them (latitude from 90 down, longitude from 0), each
    value telling where it is: wind_u = latitude, wind_v = longitude, Hs = hour, mwd = 90°."""
    hours = np.datetime64(day, "h") + np.arange(0, 24, every_hours).astype("timedelta64[h]")
    lat = np.linspace(90.0, -90.0, era5.NLAT)
    lon = np.arange(era5.NLON) * 0.25
    shape = (len(hours), era5.NLAT, era5.NLON)
    hour = (hours - np.datetime64(day, "h")).astype(float)
    np.savez_compressed(path, time=hours.astype("datetime64[s]"),
                        wind_u=np.broadcast_to(lat[None, :, None], shape).astype("f4"),
                        wind_v=np.broadcast_to(lon[None, None, :], shape).astype("f4"),
                        Hs=np.broadcast_to(hour[:, None, None], shape).astype("f4"),
                        mwd=np.full(shape, 90.0, "f4"), T_m10=np.full(shape, 8.0, "f4"))  # fmt: skip


def test_load_puts_the_grid_the_forecast_way_round_and_joins_the_days(tmp_path):
    write_day(tmp_path / "era5_20260102.npz", "2026-01-02")
    write_day(tmp_path / "era5_20260101.npz", "2026-01-01")
    fc = era5.load(sorted(tmp_path.glob("*.npz"), reverse=True))
    assert fc.lat[0] == -90.0 and fc.lat[-1] == 90.0 and fc.lon[0] == -180.0 and fc.lon[-1] == 179.75
    assert len(fc.time) == 4 and fc.time[0] == np.datetime64("2026-01-01T00:00:00")
    t = np.datetime64("2026-01-01T18:00", "s")
    assert fc.sample("wind_u", 35.1, -120.3, t) == pytest.approx(35.1, abs=1e-4)
    assert fc.sample("wind_v", 35.1, -120.3, t) == pytest.approx(360.0 - 120.3, abs=1e-3)
    assert fc.sample("Hs", 35.1, -120.3, t) == pytest.approx(6.0)  # halfway from 12 h (12) to 24 h (0 of the next day)
    assert fc.sample("wave_from_x", 0.0, 0.0, t) == pytest.approx(1.0)  # from the east
    assert fc.sample("wave_from_y", 0.0, 0.0, t) == pytest.approx(0.0, abs=1e-6)


def test_overlapping_days_are_refused(tmp_path):
    write_day(tmp_path / "a.npz", "2026-01-01")
    write_day(tmp_path / "b.npz", "2026-01-01")
    with pytest.raises(ValueError):
        era5.load([tmp_path / "a.npz", tmp_path / "b.npz"])


def test_currents_are_added_at_the_same_times(tmp_path):
    write_day(tmp_path / "era5_20260101.npz", "2026-01-01")
    hours = np.datetime64("2026-01-01", "h") + np.arange(0, 24, 12).astype("timedelta64[h]")
    shape = (2, era5.NLAT, era5.NLON)
    np.savez_compressed(tmp_path / "cmems_20260101.npz", time=hours.astype("datetime64[s]"),
                        current_u=np.full(shape, 0.5, "f4"), current_v=np.full(shape, -0.25, "f4"))
    fc = era5.load([tmp_path / "era5_20260101.npz"], currents=[tmp_path / "cmems_20260101.npz"])
    t = np.datetime64("2026-01-01T06:00", "s")
    assert fc.sample("current_u", 10.0, 20.0, t) == pytest.approx(0.5)
    assert fc.sample("current_v", 10.0, 20.0, t) == pytest.approx(-0.25)
    np.savez_compressed(tmp_path / "cmems_20260101.npz", time=hours[:1].astype("datetime64[s]"),
                        current_u=np.full((1, *shape[1:]), 0.5, "f4"), current_v=np.full((1, *shape[1:]), 0.5, "f4"))
    with pytest.raises(ValueError):
        era5.load([tmp_path / "era5_20260101.npz"], currents=[tmp_path / "cmems_20260101.npz"])


def test_currents_are_averaged_onto_the_era5_grid():
    from shiprouting.weather.cmems import to_era5_grid

    lat = np.arange(-80.0, 90.0 + 1e-9, 1 / 12)
    lon = np.arange(-180.0, 180.0 - 1e-9, 1 / 12)
    u = np.broadcast_to(lon[None, None, :], (1, len(lat), len(lon))).copy()
    u[0, :, np.abs(lon - 10.0) < 0.01] = np.nan  # the model column at 10°E is land
    out = to_era5_grid(lat, lon, u)  # ARCO layout: latitude from 90 down, longitude from 0
    row = 4 * 60  # 30°N
    assert out[0, row, 4 * 20] == pytest.approx(20.0, abs=1e-4)  # 20°E: mean of 19.917, 20, 20.083
    assert out[0, row, 4 * 10] == pytest.approx(10.0, abs=1e-4)  # 10°E: its own column left out, mean of the two beside it
    assert np.isnan(out[0, era5.NLAT - 1, 0])  # 90°S: the model does not reach
