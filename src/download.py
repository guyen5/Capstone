"""Download daily ETF prices from Yahoo Finance and freeze them (WBS 2.2).

I download once and then only load the frozen copy. Yahoo recalculates
adjusted prices after every dividend, so two downloads on different days
never match exactly; the frozen file and its checksum keep every result
reproducible.
"""

import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# Yahoo column names mapped to the snake_case names used in this project.
COLUMN_NAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
    "Capital Gains": "capital_gains",
}

# Final column order of the long table.
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close",
           "volume", "dividends", "splits", "capital_gains"]

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
ACTION_COLUMNS = ["dividends", "splits", "capital_gains"]


def _to_long(raw, tickers):
    """Turn the wide yfinance result into one row per (date, ticker).

    yf.download returns columns with two levels, field and ticker, for
    example ("Adj Close", "SPY"). The order of the two levels has changed
    between yfinance versions, so we find the ticker level by its contents
    instead of assuming its position.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected two column levels (field, ticker) from yfinance.")

    level_values = [set(raw.columns.get_level_values(i)) for i in range(2)]
    ticker_level = 0 if level_values[0] & set(tickers) else 1

    frames = []
    for ticker in tickers:
        if ticker not in level_values[ticker_level]:
            continue
        one = raw.xs(ticker, axis=1, level=ticker_level).copy()
        one = one.rename(columns=COLUMN_NAMES)
        one["ticker"] = ticker
        one["date"] = one.index
        frames.append(one)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    long = pd.concat(frames, ignore_index=True)

    # Corporate action columns only appear when a ticker had an action.
    for col in ACTION_COLUMNS:
        if col not in long.columns:
            long[col] = 0.0
        long[col] = long[col].fillna(0.0)

    # Dates as plain calendar days without time zones.
    dates = pd.to_datetime(long["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    long["date"] = dates.dt.normalize()

    # yfinance pads tickers with empty rows on days they have no data.
    # Those rows carry no information, so drop them here.
    long = long.dropna(subset=PRICE_COLUMNS + ["volume"], how="all")

    return long[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def download_prices(tickers=None, start=None, end=None, retries=3, pause=10):
    """Download daily prices, volume and corporate actions for all tickers.

    auto_adjust=False keeps both Close (the traded price) and Adj Close
    (adjusted for splits and dividends). Returns use Adj Close; Close is
    kept for the quality checks.

    Yahoo sometimes drops a ticker when it rate limits a request, so tickers
    that come back empty are requested again, up to `retries` times, with a
    longer pause each time.
    """
    import yfinance as yf

    tickers = list(tickers or config.ALL_TICKERS)
    start = start or config.START
    end = end or config.END

    collected = []
    missing = tickers
    for attempt in range(1, retries + 1):
        raw = yf.download(
            missing,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            group_by="column",
            multi_level_index=True,
            progress=False,
            threads=True,
        )
        long = _to_long(raw, missing)

        # A ticker counts as downloaded only if it has real adjusted prices.
        received = set(long.loc[long["adj_close"].notna(), "ticker"])
        collected.append(long[long["ticker"].isin(received)])
        missing = [t for t in missing if t not in received]

        print(f"Attempt {attempt}: received {len(received)} tickers, "
              f"{len(missing)} still missing {missing if missing else ''}")
        if not missing:
            break
        time.sleep(pause * attempt)

    if missing:
        raise RuntimeError(
            f"No data for {missing} after {retries} attempts. Wait a few "
            "minutes (Yahoo rate limit) and run the cell again."
        )

    prices = pd.concat(collected, ignore_index=True)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def _md5(path):
    """MD5 checksum of a file, used to prove a frozen file was not changed."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions():
    """Versions of the libraries that produced the download."""
    versions = {"python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__}
    try:
        import yfinance as yf
        versions["yfinance"] = yf.__version__
    except ImportError:
        versions["yfinance"] = "not installed"
    return versions


def freeze(prices, raw_dir, start=None, end=None, overwrite=False):
    """Save the download as Raw Prices.csv plus Raw Prices Info.json.

    The info file records when and how the data was downloaded and an MD5
    checksum of the CSV. An existing freeze is never replaced unless
    overwrite=True.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / config.FILES["raw"]
    info_path = raw_dir / config.FILES["raw_info"]

    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"{csv_path.name} already exists. Load it with load_frozen(), or "
            "pass overwrite=True to replace it."
        )

    out = prices.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)

    per_ticker = {
        ticker: {"rows": int(len(group)),
                 "first_date": group["date"].min().strftime("%Y-%m-%d"),
                 "last_date": group["date"].max().strftime("%Y-%m-%d")}
        for ticker, group in prices.groupby("ticker")
    }
    info = {
        "file": csv_path.name,
        "md5": _md5(csv_path),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance via yfinance, auto_adjust=False, actions=True",
        "start": start or config.START,
        "end_exclusive": end or config.END,
        "tickers": sorted(prices["ticker"].unique().tolist()),
        "n_rows": int(len(prices)),
        "per_ticker": per_ticker,
        "versions": _library_versions(),
    }
    info_path.write_text(json.dumps(info, indent=2))

    print(f"Frozen {info['n_rows']:,} rows for {len(info['tickers'])} tickers")
    print(f"  data: {csv_path}")
    print(f"  md5:  {info['md5']}")
    return csv_path, info_path


def is_frozen(raw_dir):
    """True if a frozen download already exists."""
    return (Path(raw_dir) / config.FILES["raw"]).exists()


def load_frozen(raw_dir, verify=True):
    """Load Raw Prices.csv and stop if its checksum does not match.

    Returns (prices, info).
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / config.FILES["raw"]
    info = json.loads((raw_dir / config.FILES["raw_info"]).read_text())

    if verify:
        actual = _md5(csv_path)
        if actual != info["md5"]:
            raise ValueError(
                f"Checksum mismatch for {csv_path.name}: expected {info['md5']}, "
                f"got {actual}. The file was changed after freezing."
            )

    prices = pd.read_csv(csv_path, parse_dates=["date"], dtype={"ticker": str})
    print(f"Loaded {csv_path.name}: {len(prices):,} rows, "
          f"{prices['ticker'].nunique()} tickers, checksum "
          f"{'verified' if verify else 'not checked'}")
    return prices, info


# File names used before October 16, 2026. rename_old_files() moves files
# with these names out of the way so the data folders only hold current files.
_OLD_PATTERNS = ["prices_raw_*.csv", "freeze_*.json", "prices_clean_v1.*",
                 "monthly_panel_v1.*", "walk_forward_calendar_v1.*",
                 "data_dictionary_v1.csv", "quality_report_*.csv", "fill_log_v1.csv",
                 "coverage_v1.csv", "panel_checks_v1.csv", "eda_*.csv"]


def rename_old_files(dirs):
    """One-time tidy-up of files saved under the old naming scheme.

    The old frozen download (prices_raw_YYYYMMDD.csv and freeze_YYYYMMDD.json)
    is renamed to Raw Prices.csv and Raw Prices Info.json, so nothing is
    downloaded again; the checksum stays valid because the content is
    unchanged. Every other old file is rebuilt by the notebooks, so it is
    moved to data/archive. Running this again does nothing.
    """
    raw = dirs["raw"]
    old_csv = sorted(raw.glob("prices_raw_*.csv"))
    old_json = sorted(raw.glob("freeze_*.json"))
    if old_csv and old_json and not is_frozen(raw):
        info = json.loads(old_json[-1].read_text())
        old_csv[-1].rename(raw / config.FILES["raw"])
        info["file"] = config.FILES["raw"]
        (raw / config.FILES["raw_info"]).write_text(json.dumps(info, indent=2))
        old_json[-1].unlink()
        print(f"Renamed {old_csv[-1].name} to {config.FILES['raw']}")

    moved = 0
    for folder in ("raw", "processed", "reports"):
        for pattern in _OLD_PATTERNS:
            for path in dirs[folder].glob(pattern):
                shutil.move(str(path), str(dirs["archive"] / path.name))
                moved += 1
    if moved:
        print(f"Moved {moved} old files to {dirs['archive']}")
