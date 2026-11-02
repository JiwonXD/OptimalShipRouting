import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import buoys

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
