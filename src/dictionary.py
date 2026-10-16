"""Data dictionary: one entry per column of every table the project saves.

This file is the single source of truth for what each variable means. The
notebooks check every saved table against it (check_table), so the
dictionary cannot drift away from the code. The written data dictionary
document is generated from the same entries.

Fields of each entry
    table     prices_clean_v1, monthly_panel_v1 or walk_forward_calendar
    name      column name in the file
    kind      date, text, float, int, bool
    unit      unit of measurement
    role      identifier, raw, feature, label, flag, split, metadata
    meaning   what the value means, in plain words
    formula   how it is computed (P = adjusted close, V = volume)
    lo, hi    allowed range (None = no bound)
    missing   when the value may be empty
    code      function that creates it
"""

import numpy as np
import pandas as pd


def _v(table, name, kind, unit, role, meaning, formula="", lo=None, hi=None,
       missing="never", code=""):
    return dict(table=table, name=name, kind=kind, unit=unit, role=role,
                meaning=meaning, formula=formula, lo=lo, hi=hi,
                missing=missing, code=code)


D, P, C = "prices_clean_v1", "monthly_panel_v1", "walk_forward_calendar"

ENTRIES = [
    # Clean daily prices (notebook 01).
    _v(D, "date", "date", "trading day", "identifier",
       "NYSE trading day.", code="quality.clean_prices"),
    _v(D, "ticker", "text", "", "identifier",
       "ETF symbol, one of the 16 tickers in config.ALL_TICKERS.",
       code="quality.clean_prices"),
    _v(D, "open", "float", "USD", "raw", "First traded price of the day, not adjusted.",
       lo=0, code="data_pull.download_prices"),
    _v(D, "high", "float", "USD", "raw", "Highest traded price of the day, not adjusted.",
       lo=0, code="data_pull.download_prices"),
    _v(D, "low", "float", "USD", "raw", "Lowest traded price of the day, not adjusted.",
       lo=0, code="data_pull.download_prices"),
    _v(D, "close", "float", "USD", "raw",
       "Last traded price of the day, not adjusted. Used only in quality checks.",
       lo=0, code="data_pull.download_prices"),
    _v(D, "adj_close", "float", "USD", "raw",
       "Close adjusted for dividends and splits, so its changes are total returns. "
       "Every return and price feature uses this column.",
       lo=0, code="data_pull.download_prices"),
    _v(D, "volume", "float", "shares", "raw",
       "Number of shares traded that day.", lo=0,
       missing="only on days filled from the previous day (none in freeze 20261016)",
       code="data_pull.download_prices"),
    _v(D, "dividends", "float", "USD per share", "raw",
       "Cash distribution with its ex-date on this day; 0 otherwise.", lo=0,
       code="data_pull.download_prices"),
    _v(D, "splits", "float", "ratio", "raw", "Split ratio on split days; 0 otherwise.",
       lo=0, code="data_pull.download_prices"),
    _v(D, "capital_gains", "float", "USD per share", "raw",
       "Capital gains distribution on this day; 0 otherwise.", lo=0,
       code="data_pull.download_prices"),
    _v(D, "filled", "bool", "", "flag",
       "True if the price was copied from the previous trading day.",
       code="quality.clean_prices"),
    _v(D, "zero_volume", "bool", "", "flag",
       "True if no shares traded that day (JPXN: 30 days in freeze 20261016).",
       code="quality.clean_prices"),

    # Monthly panel (notebook 02).
    _v(P, "date", "date", "month-end", "identifier",
       "Decision date t: last trading day of the month. Features are measured at t.",
       code="features.build_panel"),
    _v(P, "universe", "text", "", "identifier",
       "ETF_A (8 ETFs, primary) or ETF_B (10 ETFs, robustness). Ranks and labels "
       "are computed within the universe.", code="features.build_panel"),
    _v(P, "ticker", "text", "", "identifier",
       "ETF symbol. One-hot encoded at model time as the 'ticker' input.",
       code="features.build_panel"),
    _v(P, "holding_month", "text", "YYYY-MM", "identifier",
       "Month whose return the label measures (the month after t).",
       code="features.build_panel"),
    _v(P, "sample", "text", "", "split",
       "pre_test: holding month before 2020-01 (training and validation only). "
       "test: holding month 2020-01 to 2026-09. unlabeled: newest month, no label yet.",
       code="features.build_panel"),
    _v(P, "ret_1m", "float", "log return", "feature",
       "Momentum: return over the last month.", "ln(P_t / P_(t-21))",
       lo=-2, hi=2, code="features._one_ticker_features"),
    _v(P, "trend", "float", "ratio - 1", "feature",
       "Short-term trend: 10-day average relative to 50-day average.",
       "SMA10(P) / SMA50(P) - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(P, "price_bias", "float", "ratio - 1", "feature",
       "How stretched today's price is from its 10-day average.",
       "P_t / SMA10(P) - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(P, "rsi14", "float", "index 0 to 100", "feature",
       "Relative Strength Index: balance of recent gains and losses.",
       "100 - 100 / (1 + avg gain / avg loss), Wilder smoothing alpha = 1/14",
       lo=0, hi=100, code="features.wilder_rsi"),
    _v(P, "macd_diff", "float", "fraction of price", "feature",
       "MACD histogram scaled by price: is short-term momentum speeding up or slowing?",
       "[(EMA12(P) - EMA26(P)) - EMA9(EMA12 - EMA26)] / P_t",
       lo=-1, hi=1, code="features._one_ticker_features"),
    _v(P, "boll_width", "float", "fraction of price", "feature",
       "Bollinger band width: recent volatility relative to price.",
       "4 x std20(P) / SMA20(P), population std", lo=0, hi=2,
       code="features._one_ticker_features"),
    _v(P, "volume_bias", "float", "ratio - 1", "feature",
       "Trading activity today relative to its 10-day average.",
       "V_t / SMA10(V) - 1; set to 0 when V_t = 0 (see volume_bias_missing)",
       lo=-1, code="features._one_ticker_features"),
    _v(P, "volume_bias_missing", "bool", "", "flag",
       "True if volume bias was undefined at t (zero volume) and filled with 0.",
       code="features.build_panel"),
    _v(P, "next_return", "float", "simple return", "label",
       "Label A: return from month-end t to the next month-end. Target of Model A "
       "and the return used in portfolio accounting.", "P_(t+1) / P_t - 1",
       lo=-1, missing="newest month only (sample = unlabeled)",
       code="features._universe_rows"),
    _v(P, "next_rank", "int", "rank", "label",
       "Rank of next_return within the universe that month; 1 = best. Used to "
       "redefine the label as top 2 or top 4 in robustness checks.",
       lo=1, hi=10, missing="newest month only", code="features._universe_rows"),
    _v(P, "top3_next", "int", "0 or 1", "label",
       "Label B: 1 if the ETF finishes in next month's top 3 of its universe. "
       "Target of Model B.", "1 if next_rank <= 3 else 0", lo=0, hi=1,
       missing="newest month only", code="features.build_panel"),

    # Walk-forward calendar (notebook 02, used from notebook 03 on).
    _v(C, "holding_month", "text", "YYYY-MM", "identifier",
       "Test month whose return is predicted.", code="splits.walk_forward_calendar"),
    _v(C, "decision_date", "date", "month-end", "identifier",
       "Month-end t when the model is refit and predicts.",
       code="splits.walk_forward_calendar"),
    _v(C, "train_start", "date", "month-end", "split",
       "First panel date in the training window.", code="splits.walk_forward_calendar"),
    _v(C, "train_end", "date", "month-end", "split",
       "Last panel date in the training window; always before decision_date.",
       code="splits.walk_forward_calendar"),
    _v(C, "val_start", "date", "month-end", "split",
       "First date of the 12-month validation slice inside the training window.",
       code="splits.walk_forward_calendar"),
    _v(C, "n_train_months", "int", "months", "split",
       "Number of month-ends in the training window, validation included.",
       lo=1, code="splits.walk_forward_calendar"),
    _v(C, "window", "text", "", "split",
       "growing (primary) or 36 / 60 (rolling robustness windows).",
       code="splits.walk_forward_calendar"),
]


def table_entries(table):
    """Dictionary entries of one table, in column order."""
    return [e for e in ENTRIES if e["table"] == table]


def to_frame():
    """The whole dictionary as a DataFrame (saved as data_dictionary_v1.csv)."""
    return pd.DataFrame(ENTRIES)


_KIND_CHECK = {
    "date": pd.api.types.is_datetime64_any_dtype,
    "text": lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
    "float": pd.api.types.is_float_dtype,
    "int": pd.api.types.is_integer_dtype,
    "bool": pd.api.types.is_bool_dtype,
}


def check_table(df, table):
    """Compare a saved table with its dictionary entries.

    Checks: same columns in the same order, the declared type, the allowed
    range, and no empty values where the dictionary says "never".
    Returns a report DataFrame with one row per column.
    """
    entries = table_entries(table)
    expected = [e["name"] for e in entries]
    rows = [{"check": "columns", "column": "ALL",
             "status": "PASS" if list(df.columns) == expected else "FAIL",
             "detail": "same columns and order as the dictionary" if list(df.columns) == expected
             else f"table {list(df.columns)} vs dictionary {expected}"}]

    for e in entries:
        name = e["name"]
        if name not in df.columns:
            continue
        s = df[name]
        problems = []
        if not _KIND_CHECK[e["kind"]](s):
            problems.append(f"type {s.dtype}, expected {e['kind']}")
        n_na = int(s.isna().sum())
        if e["missing"] == "never" and n_na:
            problems.append(f"{n_na} empty values")
        values = s.dropna()
        if e["kind"] in ("float", "int") and len(values):
            if e["lo"] is not None and values.min() < e["lo"]:
                problems.append(f"min {values.min():.4g} below {e['lo']}")
            if e["hi"] is not None and values.max() > e["hi"]:
                problems.append(f"max {values.max():.4g} above {e['hi']}")
        detail = "; ".join(problems) if problems else (
            f"{e['kind']}, {n_na} empty" + (
                f", range {values.min():.4g} to {values.max():.4g}"
                if e["kind"] in ("float", "int") and len(values) else ""))
        rows.append({"check": "column", "column": name,
                     "status": "FAIL" if problems else "PASS", "detail": detail})
    return pd.DataFrame(rows)
