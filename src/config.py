"""Settings used by every notebook: tickers, dates, windows, file names."""

from pathlib import Path


# Universes from Wang and Hasuike (2026). ETF_A is the main test universe,
# ETF_B the robustness check. SPY and XLK are in both.
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
TOP_K = 3                    # top-3 label, kept from the earlier design; not used in the main test

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

# Daily rows (Notebook 3). Each trading day's features are paired with the
# return over the next 21 trading days, about one month. A row can be used
# for training only once those 21 days have passed.
FORWARD_DAYS = 21

# Daily training windows: name -> (total months, validation months).
# "9+3" is Wang and Hasuike's window (9 months of training, 3 of
# validation). 36 and 60 months are the longer candidates; the one used in
# the main test is chosen on 2015 to 2019 data.
DAILY_WINDOWS = {"9+3": (12, 3), "36": (36, 12), "60": (60, 12)}

# Covariance: 220 trading days ending at the decision date, scaled from
# daily to monthly by 21 days. The paper adds 1e-6 to the diagonal of the
# sample matrix; the robust model uses Ledoit-Wolf shrinkage instead.
COV_DAYS = 220
COV_RIDGE = 1e-6
MONTH_DAYS = 21

# Experiment settings, fixed before the main test (Project Brief, section 7).
FEE = 0.005                                     # charged on value traded
LAMBDA_GRID = (0.1, 1, 10, 20, 50)              # risk aversion, the paper's grid
KAPPA_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)  # trading penalty in the optimizer
TAU_GRID = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)  # Black-Litterman prior uncertainty
BOOTSTRAP_REFITS = 100                          # refits per forecast for standard errors
BOOTSTRAP_BLOCK_MONTHS = 3                      # months per resampled block
SELECTION_PERIOD = ("2015-01", "2019-12")       # holding months used to choose settings
SECOND_TEST_PERIOD = ("2011-01", "2019-12")     # robustness test period

# Replication (Replication Spec). Table 3 turnover values are multiples of
# 1/70, so results are also compared over the first 70 test months.
PAPER_LAST_MONTH = "2025-10"
RIDGE_ALPHAS = tuple(float(f"{a:.3g}") for a in [10 ** (k / 2) for k in range(-8, 7)])  # 1e-4 to 1e3, 15 values

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
    "daily_table": "Daily Features.parquet",
    "daily_windows": "Daily Training Windows.parquet",
    "covariances": "Covariances.parquet",
    "daily_info": "Daily Features Info.json",
    "daily_checks": "Daily Data Checks.csv",
    "benchmark_months": "Benchmark Months.parquet",
    "benchmark_results": "Benchmark Results.csv",
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
