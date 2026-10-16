"""Settings used by every notebook: tickers, dates, windows, file names."""

from pathlib import Path


# Universes from Wang and Hasuike (2026). ETF_A is my primary test,
# ETF_B my robustness check. SPY and XLK are in both.
ETF_A = ["SPY", "VTI", "XLK", "EFA", "EEM", "JPXN", "AGG", "DBC"]
ETF_B = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"]
ALL_TICKERS = sorted(set(ETF_A) | set(ETF_B))

# Download window. The end date is exclusive in yfinance, so the last day
# kept is September 30, 2026. A fixed end date means every re-download
# covers the same days.
START = "2009-01-01"
END = "2026-10-01"
EXPECTED_FIRST_DAY = "2009-01-02"
YFINANCE_VERSION = "1.7.0"

# Quality check thresholds.
MAX_DAILY_MOVE = 0.25        # flag one-day moves above 25%
FFILL_LIMIT = 1              # fill at most one missing day in a row
STALE_RUN = 5                # flag 5 or more identical closes in a row
ADJ_RATIO_TOL = 1e-4         # tolerance for Close vs Adj Close changes
CROSS_CHECK_TOL = 0.001      # 0.1 percentage point on monthly returns

# Monthly data. A row dated t (a month-end) holds features known at t and
# the return earned from t to the next month-end.
PANEL_START = "2010-01-01"
TOP_K = 3

# Feature windows in trading days.
RET_WINDOW = 21
SMA_SHORT = 10
SMA_LONG = 50
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLL_WINDOW = 20

# The seven model inputs, in a fixed order.
FEATURES = ["momentum", "trend", "price_bias", "rsi", "macd", "bollinger", "volume_bias"]

# Test period, as the months whose returns are predicted.
TEST_FIRST_MONTH = "2020-01"
TEST_LAST_MONTH = "2026-09"
VALIDATION_MONTHS = 12
ROLLING_WINDOWS = (36, 60)

# Google Drive folders and file names.
DRIVE_ROOT = Path("/content/drive/MyDrive/Capstone")
FILES = {
    "raw": "Raw Prices.csv",
    "raw_info": "Raw Prices Info.json",
    "daily": "Daily Prices.parquet",
    "monthly": "Monthly Data.parquet",
    "monthly_info": "Monthly Data Info.json",
    "windows": "Training Windows.parquet",
    "variables": "Variables.csv",
    "quality": "Quality Report.csv",
    "filled": "Filled Prices.csv",
    "coverage": "Coverage.csv",
    "checks": "Monthly Data Checks.csv",
}


def data_dirs(root=None):
    """Return the raw, processed and reports folders, creating them if needed."""
    root = Path(root or DRIVE_ROOT)
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "reports": root / "data" / "reports",
        "archive": root / "data" / "archive",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
