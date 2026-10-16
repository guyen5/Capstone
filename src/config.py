"""Project settings shared by every notebook.

Every notebook imports its tickers, dates, thresholds and folder paths from
this file. To change a setting, change it here and nowhere else, so all
steps of the pipeline always agree with each other.
"""

from pathlib import Path


# Asset universes, taken from Wang and Hasuike (2026).
# ETF_A is the primary universe, ETF_B is the robustness universe.
ETF_A = ["SPY", "VTI", "XLK", "EFA", "EEM", "JPXN", "AGG", "DBC"]
ETF_B = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"]

# The 16 unique tickers we download (SPY and XLK are in both universes).
ALL_TICKERS = sorted(set(ETF_A) | set(ETF_B))

# Download window.
# START gives one full year of history before the first modelling row
# (January 2010), which every indicator lookback needs.
# END is exclusive in yfinance, so "2026-10-01" means the last day kept is
# Wednesday, September 30, 2026: the last complete month of the test period.
# Fixing END (instead of "today") makes every re-download cover the same dates.
START = "2009-01-01"
END = "2026-10-01"

# The first trading day of 2009 was Friday, January 2. Every ticker must
# start on or before this date.
EXPECTED_FIRST_DAY = "2009-01-02"

# yfinance version recorded in the freeze metadata and pinned in Colab.
YFINANCE_VERSION = "1.7.0"

# Quality check thresholds (see section 2.9 of the data specification).
MAX_DAILY_MOVE = 0.25        # flag one-day adjusted moves larger than 25%
FFILL_LIMIT = 1              # fill at most one missing trading day in a row
STALE_RUN = 5                # flag 5 or more identical closes in a row
ADJ_RATIO_TOL = 1e-4         # tolerance for Close vs Adj Close ratio changes
CROSS_CHECK_TOL = 0.001      # 0.1 percentage point on monthly returns

# Monthly modelling table (notebook 02).
# A row dated t (a month-end) holds features known at t and the return
# earned while holding from t to the next month-end.
PANEL_START = "2010-01-01"   # first decision month-end kept in the panel
TOP_K = 3                    # label B: is the ETF in next month's top 3?

# Feature windows in trading days (section 2.5 of the data specification).
RET_WINDOW = 21              # 1-month log return
SMA_SHORT = 10               # short moving average (trend, price bias, volume)
SMA_LONG = 50                # long moving average (trend)
RSI_WINDOW = 14              # Wilder RSI
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLL_WINDOW = 20             # Bollinger band width

# The seven model inputs, in a fixed order used by every notebook.
FEATURES = ["ret_1m", "trend", "price_bias", "rsi14",
            "macd_diff", "boll_width", "volume_bias"]

# Walk-forward test period, written as holding months.
# The first prediction is made at the end of December 2019 and earns the
# January 2020 return; the last earns the September 2026 return.
TEST_FIRST_HOLDING = "2020-01"
TEST_LAST_HOLDING = "2026-09"
VALIDATION_MONTHS = 12       # last 12 training months are used to tune settings
ROLLING_WINDOWS = (36, 60)   # robustness: rolling 3 and 5 year windows

# Folders on Google Drive. Colab mounts Drive at /content/drive.
DRIVE_ROOT = Path("/content/drive/MyDrive/Capstone")


def data_dirs(root=None):
    """Return the raw, processed and reports folders, creating them if needed.

    raw/        frozen downloads, never edited by hand
    processed/  cleaned tables built from raw/ by code
    reports/    quality reports and coverage tables
    """
    root = Path(root or DRIVE_ROOT)
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "reports": root / "data" / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
