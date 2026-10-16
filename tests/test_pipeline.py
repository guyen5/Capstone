"""Tests for the data pipeline (WBS 2.2 to 3.1) on simulated prices.

Run from the repository root:
    python -m pytest tests

No internet is needed: yfinance is replaced by a function that returns
simulated prices in exactly the format Yahoo uses. Known problems are
planted in the simulated data (a missing day, zero-volume days, a price
spike) to check that the quality checks catch them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, dictionary, download, features, quality, walk_forward  # noqa: E402

DATES = pd.bdate_range("2009-01-02", "2026-09-30")


def fake_yahoo(tickers, start=None, end=None, **kwargs):
    """Simulated yf.download output: columns (field, ticker), daily rows."""
    rng = np.random.default_rng(0)
    fields = ["Adj Close", "Capital Gains", "Close", "Dividends", "High", "Low",
              "Open", "Stock Splits", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    df = pd.DataFrame(np.nan, index=pd.DatetimeIndex(DATES, name="Date"), columns=cols)
    n = len(DATES)
    for t in tickers:
        close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        div = np.zeros(n)
        div[60::63] = 0.3
        factor = np.ones(n)
        for i in np.where(div > 0)[0]:
            factor[:i] *= 1 - div[i] / close[i - 1]
        opn = close * (1 + rng.normal(0, 0.002, n))
        df[("Close", t)] = close
        df[("Adj Close", t)] = close * factor
        df[("Open", t)] = opn
        df[("High", t)] = np.maximum(opn, close) * 1.003
        df[("Low", t)] = np.minimum(opn, close) * 0.997
        df[("Volume", t)] = rng.integers(100_000, 10_000_000, n).astype(float)
        df[("Dividends", t)] = div
        df[("Stock Splits", t)] = 0.0
        df[("Capital Gains", t)] = 0.0
    if "JPXN" in tickers:
        df.loc[DATES[100], (slice(None), "JPXN")] = np.nan     # one missing day
        df.loc[DATES[200:203], ("Volume", "JPXN")] = 0          # zero volume
    if "DBC" in tickers:
        df.loc[DATES[500], ("Adj Close", "DBC")] *= 1.4         # price spike
    return df


@pytest.fixture(scope="module")
def raw(monkeypatch_module):
    import yfinance as yf
    monkeypatch_module.setattr(yf, "download", fake_yahoo)
    return download.download_prices(pause=0)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def clean(raw):
    return quality.clean_prices(raw)[0]


@pytest.fixture(scope="module")
def panel(clean):
    return features.build_panel(clean)


def test_download_shape(raw):
    assert raw["ticker"].nunique() == 16
    assert list(raw.columns) == download.COLUMNS


def test_quality_checks_catch_planted_problems(raw):
    report = quality.run_quality_checks(raw)
    warn = report[report["status"] == "WARN"]
    assert ("missing days", "JPXN") in set(zip(warn["check"], warn["ticker"]))
    assert ("zero volume", "JPXN") in set(zip(warn["check"], warn["ticker"]))
    assert ("large moves", "DBC") in set(zip(warn["check"], warn["ticker"]))
    assert not (report["status"] == "FAIL").any()


def test_freeze_roundtrip(raw, tmp_path):
    csv_path, _ = download.freeze(raw, tmp_path)
    loaded, meta = download.load_frozen(tmp_path)
    assert len(loaded) == len(raw) == meta["n_rows"]
    with pytest.raises(FileExistsError):
        download.freeze(raw, tmp_path)
    csv_path.write_text(csv_path.read_text() + "tampered")
    with pytest.raises(ValueError):
        download.load_frozen(tmp_path)


def test_features_match_hand_calculation(clean):
    daily = features.daily_features(clean)
    spy = clean[clean["ticker"] == "SPY"].set_index("date")
    p, v = spy["adj_close"], spy["volume"].astype(float)
    i = 3000
    row = daily[(daily["ticker"] == "SPY") & (daily["date"] == p.index[i])].iloc[0]

    def ema(x, span):
        a, out = 2 / (span + 1), [x[0]]
        for z in x[1:]:
            out.append(a * z + (1 - a) * out[-1])
        return np.array(out)

    ch = p.diff().iloc[1:i + 1].to_numpy()
    g, lo = np.clip(ch, 0, None), np.clip(-ch, 0, None)
    ag, al = g[0], lo[0]
    for k in range(1, len(ch)):
        ag, al = ag + (g[k] - ag) / 14, al + (lo[k] - al) / 14
    x = p.iloc[:i + 1].to_numpy()
    macd = ema(x, 12) - ema(x, 26)

    hand = {
        "ret_1m": np.log(p.iloc[i] / p.iloc[i - 21]),
        "trend": p.iloc[i - 9:i + 1].mean() / p.iloc[i - 49:i + 1].mean() - 1,
        "price_bias": p.iloc[i] / p.iloc[i - 9:i + 1].mean() - 1,
        "rsi14": 100 - 100 / (1 + ag / al),
        "macd_diff": (macd[-1] - ema(macd, 9)[-1]) / x[-1],
        "boll_width": 4 * p.iloc[i - 19:i + 1].std(ddof=0) / p.iloc[i - 19:i + 1].mean(),
        "volume_bias": v.iloc[i] / v.iloc[i - 9:i + 1].mean() - 1,
    }
    for name, value in hand.items():
        assert row[name] == pytest.approx(value, abs=1e-9), name


def test_panel_passes_all_checks(panel, clean):
    report = features.validate_panel(panel, clean)
    assert (report["status"] == "PASS").all(), report[report["status"] != "PASS"]
    assert (dictionary.check_table(panel, "monthly_panel_v1")["status"] == "PASS").all()
    assert (dictionary.check_table(clean, "prices_clean_v1")["status"] == "PASS").all()


def test_zero_volume_month_end_is_flagged(clean):
    c = clean.copy()
    d = features.month_ends(c)[80]
    c.loc[(c["ticker"] == "JPXN") & (c["date"] == d), "volume"] = 0
    p = features.build_panel(c)
    row = p[(p["ticker"] == "JPXN") & (p["date"] == d)].iloc[0]
    assert row["volume_bias"] == 0 and row["volume_bias_missing"]


def test_walk_forward_calendar(panel):
    for window in ["growing", *config.ROLLING_WINDOWS]:
        cal = walk_forward.walk_forward_calendar(panel, window)
        assert len(cal) == 81
        assert (walk_forward.check_calendar(cal, panel)["status"] == "PASS").all()
        assert (dictionary.check_table(cal, "walk_forward_calendar")["status"] == "PASS").all()
