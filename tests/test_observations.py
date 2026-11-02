import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import observations as buoys

REALTIME = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 09 20 01 00 250  7.0  9.0   2.1   9.0   6.2 270 1012.0  18.0  19.0  15.0   MM   MM    MM
2026 09 20 00 50 250  6.0  8.0    MM    MM    MM  MM 1012.0  18.0  19.0  15.0   MM   MM    MM
"""
YEARLY = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  mi    ft
2025 12 01 00 00 999 99.0 99.0  3.40  12.0   8.1 999 1020.0 999.0 999.0 999.0 99.0 99.00
"""


def test_stdmet_files_are_read_in_time_order_with_missing_values_as_nan():
    o = buoys.read_stdmet(REALTIME)
    assert list(o["time"].astype(str)) == ["2026-09-20T00:50:00", "2026-09-20T01:00:00"]
    assert np.isnan(o["WVHT"][0]) and o["WVHT"][1] == 2.1 and o["WSPD"][1] == 7.0
    y = buoys.read_stdmet(YEARLY)
    assert np.isnan(y["WSPD"][0]) and y["WVHT"][0] == 3.4


def test_station_table_locations():
    table = ("# STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD | LOCATION | TIMEZONE | FORECAST | NOTE\n"
             "46006|NDBC|Weather Buoy|6N|SOUTHEAST PAPA|SCOOP|40.782 N 137.397 W (40&#176;46'55\" N 137&#176;23'49\" W)| | |\n")
    assert buoys.stations(table) == {"46006": (40.782, -137.397)}


def test_scores():
    s = buoys.scores(np.array([2.0, 3.0, np.nan]), np.array([1.0, 2.0, 5.0]))
    assert s["n"] == 2 and s["bias"] == pytest.approx(1.0) and s["rmse"] == pytest.approx(1.0) and s["si"] == pytest.approx(0.0)


def test_altimeter_blocks_are_whole_and_averaged_across_the_dateline():
    t = np.datetime64("2026-09-01T00:00:00", "s") + np.arange(14).astype("timedelta64[s]")
    lon = np.r_[np.linspace(179.5, 179.9, 6), np.linspace(-179.9, -179.5, 6), [0.0, 0.0]]
    hs = np.r_[np.full(6, 2.0), np.full(5, 3.0), [np.nan], [1.0, 1.0]]
    b = buoys.blocks(t, np.zeros(14), lon, hs, seconds=6)
    assert len(b["time"]) == 1  # the second block lost a point, the third is short
    assert b["time"][0] == np.datetime64("2026-09-01T00:00:03") and b["Hs"][0] == 2.0
    assert b["lon"][0] == pytest.approx(179.7, abs=1e-9)
