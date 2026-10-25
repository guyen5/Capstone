"""Training windows for the walk-forward test (WBS 3.1 and 3.2).

At each decision date t, the model is refit on rows whose labels are already
known on t and predicts the coming month.

Monthly windows (Notebook 2) are for models fitted on the monthly table: the
last 12 months of each window are the validation slice; "growing" uses every
month since 2010, 36 and 60 only the most recent 3 or 5 years.

Daily windows (Notebook 3) are for the replication and the robust model,
which train on daily rows: "9+3" is Wang and Hasuike's window, 36 and 60
months are the longer candidates (config.DAILY_WINDOWS).
"""

import pandas as pd

from src import config


def training_windows(panel, window="growing"):
    """One row per test month: decision date, training window, validation start."""
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    test = panel[panel["period"] == "test"]
    decisions = pd.DatetimeIndex(sorted(test["date"].unique()))

    rows = []
    for t in decisions:
        usable = dates[dates < t]
        if window != "growing":
            usable = usable[-int(window):]
        rows.append({
            "month": str(t.to_period("M") + 1),
            "decision_date": t,
            "train_start": usable[0],
            "train_end": usable[-1],
            "validation_start": usable[-config.VALIDATION_MONTHS],
            "train_months": len(usable),
            "window": str(window),
        })
    return pd.DataFrame(rows)


def rows_for(panel, step, universe):
    """Split the panel for one calendar row into train, validation and test.

    step: one row of training_windows(). Returns three DataFrames.
    The fit slice is train minus validation; after tuning, refit on train.
    """
    p = panel[panel["universe"] == universe]
    train = p[(p["date"] >= step["train_start"]) & (p["date"] <= step["train_end"])]
    val = train[train["date"] >= step["validation_start"]]
    test = p[p["date"] == step["decision_date"]]
    return train, val, test


def check_windows(calendar, panel):
    """Checks that no training or validation row can see the test month."""
    rows = []
    ok = (calendar["train_end"] < calendar["decision_date"]).all()
    rows.append({"check": "train ends before decision", "status": "PASS" if ok else "FAIL",
                 "detail": "every training window ends at least one month-end before t"})
    first, last = calendar["month"].iloc[0], calendar["month"].iloc[-1]
    ok = (first == config.TEST_FIRST_MONTH) and (last == config.TEST_LAST_MONTH)
    rows.append({"check": "test period", "status": "PASS" if ok else "FAIL",
                 "detail": f"{len(calendar)} test months, {first} to {last}"})
    ok = calendar["month"].is_unique
    rows.append({"check": "one prediction per month", "status": "PASS" if ok else "FAIL",
                 "detail": "no test month repeated"})
    labels_known = panel[panel["date"].isin(calendar["train_end"])]["next_return"].notna().all()
    rows.append({"check": "training labels known", "status": "PASS" if labels_known else "FAIL",
                 "detail": "the newest training row of every window has a label"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Daily training windows (Notebook 3, WBS 3.2)
#
# For daily rows, a row dated d is usable at decision date t only once its
# 21-day target is known: label_date <= t. A window of M months ends at the
# newest usable row; its last V months are the validation slice. Training
# rows must also have their target known before validation starts, which
# leaves a gap of 21 trading days so that no training target overlaps the
# validation period.
# ---------------------------------------------------------------------------

def daily_windows(table, name, total_months=None, val_months=None):
    """One row per test month: decision date and the daily rows to use.

    name: a key of config.DAILY_WINDOWS ("9+3", "36" or "60"), or any label
    when total_months and val_months are given.
    Columns: month, decision_date, train_start, fit_end, validation_start,
    train_end, train_months, validation_months, window.
    """
    if total_months is None:
        total_months, val_months = config.DAILY_WINDOWS[name]
    t = table.drop_duplicates("date")[["date", "label_date", "month_end"]].sort_values("date")
    dates = pd.DatetimeIndex(t["date"])
    label = pd.Series(t["label_date"].to_numpy(), index=dates)

    first = pd.Period(config.TEST_FIRST_MONTH, "M")
    last = pd.Period(config.TEST_LAST_MONTH, "M")
    decisions = [d for d in t.loc[t["month_end"], "date"]
                 if first <= d.to_period("M") + 1 <= last]

    rows = []
    for d in decisions:
        usable = dates[(label <= d).to_numpy()]
        train_end = usable[-1]
        start_cut = train_end - pd.DateOffset(months=total_months)
        val_cut = train_end - pd.DateOffset(months=val_months)
        train_start = usable[usable > start_cut][0]
        validation_start = usable[usable > val_cut][0]
        # Training rows whose targets are known before validation starts.
        fit = usable[(usable >= train_start)
                     & (label.reindex(usable) < validation_start).to_numpy()]
        rows.append({
            "month": str(d.to_period("M") + 1),
            "decision_date": d,
            "train_start": train_start,
            "fit_end": fit[-1],
            "validation_start": validation_start,
            "train_end": train_end,
            "train_months": total_months,
            "validation_months": val_months,
            "window": name,
        })
    return pd.DataFrame(rows)


def daily_rows_for(table, step, universe):
    """Split the daily table for one window row into fit, validation and test rows.

    fit: rows from train_start to fit_end (targets known before validation).
    validation: rows from validation_start to train_end.
    test: the month-end row at the decision date.
    After tuning, the model is refit on fit and validation rows together.
    """
    p = table[table["universe"] == universe]
    fit = p[(p["date"] >= step["train_start"]) & (p["date"] <= step["fit_end"])]
    val = p[(p["date"] >= step["validation_start"]) & (p["date"] <= step["train_end"])]
    test = p[p["date"] == step["decision_date"]]
    return fit, val, test


def check_daily_windows(calendar, table):
    """Checks that no fit or validation row can see the test month."""
    label = table.drop_duplicates("date").set_index("date")["label_date"]
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    known = (label.reindex(calendar["train_end"]).to_numpy()
             <= calendar["decision_date"].to_numpy()).all()
    add("targets known at decision", bool(known),
        "the newest training row's target is known on the decision date")
    gap = (label.reindex(calendar["fit_end"]).to_numpy()
           < calendar["validation_start"].to_numpy()).all()
    add("gap before validation", bool(gap),
        "every fit target is known before the validation slice starts")
    order = ((calendar["train_start"] <= calendar["fit_end"])
             & (calendar["fit_end"] < calendar["validation_start"])
             & (calendar["validation_start"] <= calendar["train_end"])
             & (calendar["train_end"] < calendar["decision_date"])).all()
    add("dates in order", bool(order), "train start < fit end < validation < train end < decision")
    first, last = calendar["month"].iloc[0], calendar["month"].iloc[-1]
    add("test period", first == config.TEST_FIRST_MONTH and last == config.TEST_LAST_MONTH
        and calendar["month"].is_unique, f"{len(calendar)} test months, {first} to {last}")
    return pd.DataFrame(rows)
