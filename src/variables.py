"""Definition of every column I save (WBS 2.5).

This list is the single source for what each variable means. Notebook 2
checks every saved table against it with check_table(), and the Variables
tab of the Data Specification sheet is generated from it, so the code and
the documentation always agree.
"""

import pandas as pd


def _v(table, column, label, kind, unit, meaning, formula="", lo=None, hi=None,
       empty="never", code=""):
    return dict(table=table, column=column, label=label, kind=kind, unit=unit,
                meaning=meaning, formula=formula, lo=lo, hi=hi, empty=empty, code=code)


D, M, W = "Daily Prices", "Monthly Data", "Training Windows"

ENTRIES = [
    _v(D, "date", "Date", "date", "trading day", "NYSE trading day", code="cleaning.clean_prices"),
    _v(D, "ticker", "Ticker", "text", "", "ETF symbol (16 ETFs)", code="cleaning.clean_prices"),
    _v(D, "open", "Open", "float", "USD", "First trade of the day, not adjusted", lo=0, code="download.download_prices"),
    _v(D, "high", "High", "float", "USD", "Highest trade of the day, not adjusted", lo=0, code="download.download_prices"),
    _v(D, "low", "Low", "float", "USD", "Lowest trade of the day, not adjusted", lo=0, code="download.download_prices"),
    _v(D, "close", "Close", "float", "USD", "Last trade of the day, not adjusted. Only used in quality checks", lo=0, code="download.download_prices"),
    _v(D, "adj_close", "Adjusted close", "float", "USD", "Close adjusted for dividends and splits. All returns and features use this", lo=0, code="download.download_prices"),
    _v(D, "volume", "Volume", "float", "shares", "Shares traded that day", lo=0, empty="only on filled days (none in my data)", code="download.download_prices"),
    _v(D, "dividends", "Dividend", "float", "USD per share", "Cash paid on the ex-dividend date, else 0", lo=0, code="download.download_prices"),
    _v(D, "splits", "Split", "float", "ratio", "Split ratio on split days, else 0", lo=0, code="download.download_prices"),
    _v(D, "capital_gains", "Capital gains", "float", "USD per share", "Capital gains paid that day, else 0", lo=0, code="download.download_prices"),
    _v(D, "filled", "Filled", "bool", "", "Price copied from the previous day", code="cleaning.clean_prices"),
    _v(D, "zero_volume", "Zero volume", "bool", "", "No shares traded that day (JPXN, 30 days)", code="cleaning.clean_prices"),

    _v(M, "date", "Decision date", "date", "month-end", "Last trading day of the month; features are measured here", code="features.build_panel"),
    _v(M, "universe", "Universe", "text", "", "ETF_A (primary, 8 ETFs) or ETF_B (robustness, 10 ETFs)", code="features.build_panel"),
    _v(M, "ticker", "Ticker", "text", "", "ETF symbol; one-hot encoded as a model input", code="features.build_panel"),
    _v(M, "month", "Holding month", "text", "YYYY-MM", "The month whose return the labels describe", code="features.build_panel"),
    _v(M, "period", "Period", "text", "", "history (2010 to 2019, training only), test (2020-01 to 2026-09) or latest (no label yet)", code="features.build_panel"),
    _v(M, "momentum", "Momentum", "float", "log return", "Return over the last month", "ln(P today / P 21 days ago)", lo=-2, hi=2, code="features._one_ticker_features"),
    _v(M, "trend", "Trend", "float", "ratio", "10-day average relative to 50-day average", "SMA10 / SMA50 - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(M, "price_bias", "Price bias", "float", "ratio", "Price relative to its 10-day average", "P / SMA10 - 1", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(M, "rsi", "RSI", "float", "0 to 100", "Balance of recent gains and losses", "100 - 100 / (1 + avg gain / avg loss), Wilder smoothing, 14 days", lo=0, hi=100, code="features.wilder_rsi"),
    _v(M, "macd", "MACD", "float", "share of price", "Whether short-term momentum is speeding up or slowing", "(MACD line - signal line) / P, with EMA 12, 26, 9", lo=-1, hi=1, code="features._one_ticker_features"),
    _v(M, "bollinger", "Bollinger width", "float", "share of price", "Recent volatility relative to price", "4 x std20 / SMA20", lo=0, hi=2, code="features._one_ticker_features"),
    _v(M, "volume_bias", "Volume bias", "float", "ratio", "Trading volume relative to its 10-day average", "V / SMA10(V) - 1, set to 0 on zero-volume days", lo=-1, code="features._one_ticker_features"),
    _v(M, "volume_filled", "Volume filled", "bool", "", "Volume bias was set to 0 because nothing traded that day", code="features.build_panel"),
    _v(M, "next_return", "Next-month return", "float", "simple return", "Label A: return over the holding month. Model A target and portfolio return", "P next month-end / P this month-end - 1", lo=-1, empty="latest month only", code="features._universe_rows"),
    _v(M, "next_rank", "Next-month rank", "int", "rank", "Rank of next-month return within the universe, 1 = best", lo=1, hi=10, empty="latest month only", code="features._universe_rows"),
    _v(M, "top3", "Top 3", "int", "0 or 1", "Label B: 1 if the ETF finishes in next month's top 3. Model B target", "1 if next rank <= 3", lo=0, hi=1, empty="latest month only", code="features.build_panel"),

    _v(W, "month", "Test month", "text", "YYYY-MM", "Month being predicted", code="windows.training_windows"),
    _v(W, "decision_date", "Decision date", "date", "month-end", "When the model is refit and predicts", code="windows.training_windows"),
    _v(W, "train_start", "Training start", "date", "month-end", "First month in the training window", code="windows.training_windows"),
    _v(W, "train_end", "Training end", "date", "month-end", "Last month in the training window, always before the decision date", code="windows.training_windows"),
    _v(W, "validation_start", "Validation start", "date", "month-end", "First of the 12 validation months at the end of the window", code="windows.training_windows"),
    _v(W, "train_months", "Training months", "int", "months", "Months in the window, validation included", lo=1, code="windows.training_windows"),
    _v(W, "window", "Window", "text", "", "growing (primary), 36 or 60 months (robustness)", code="windows.training_windows"),
]


def table_entries(table):
    """Entries of one table, in column order."""
    return [e for e in ENTRIES if e["table"] == table]


def to_frame():
    """All entries as a DataFrame (saved as Variables.csv)."""
    return pd.DataFrame(ENTRIES)


_KIND_CHECK = {
    "date": pd.api.types.is_datetime64_any_dtype,
    "text": lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
    "float": pd.api.types.is_float_dtype,
    "int": pd.api.types.is_integer_dtype,
    "bool": pd.api.types.is_bool_dtype,
}


def check_table(df, table):
    """Compare a saved table with its entries: columns, types, ranges, empties."""
    entries = table_entries(table)
    expected = [e["column"] for e in entries]
    same = list(df.columns) == expected
    rows = [{"check": "columns", "column": "all", "status": "PASS" if same else "FAIL",
             "detail": "same columns and order as Variables" if same
             else f"table {list(df.columns)} vs Variables {expected}"}]

    for e in entries:
        name = e["column"]
        if name not in df.columns:
            continue
        s = df[name]
        problems = []
        if not _KIND_CHECK[e["kind"]](s):
            problems.append(f"type {s.dtype}, expected {e['kind']}")
        n_na = int(s.isna().sum())
        if e["empty"] == "never" and n_na:
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
