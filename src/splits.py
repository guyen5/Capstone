"""Walk-forward calendar: which panel rows train, tune and test each month.

At each decision date t (a month-end in the test period) the model is
refit and then predicts the returns of the coming month. Only rows whose
labels are already known at t may be used:

    a row dated s has its label (the return from s to the next month-end)
    known at the next month-end, so it is usable at t only if s < t.

Window types
    "growing"      every usable row from PANEL_START (primary choice)
    36 or 60       only the last 36 or 60 usable month-ends (robustness)

Inside each training window the last VALIDATION_MONTHS month-ends form the
validation slice used to choose model settings. After tuning, the model is
refit on the whole window (training plus validation).
"""

import pandas as pd

from src import config


def walk_forward_calendar(panel, window="growing"):
    """One row per test month describing the training and validation windows.

    Columns
        holding_month   month whose return is predicted (e.g. 2020-01)
        decision_date   month-end when the prediction is made (t)
        train_start     first row date in the training window
        train_end       last row date in the training window (< t)
        val_start       first row date of the validation slice
        n_train_months  number of month-ends in the window, validation included
        window          "growing", 36 or 60
    """
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    test = panel[panel["sample"] == "test"]
    decisions = pd.DatetimeIndex(sorted(test["date"].unique()))

    rows = []
    for t in decisions:
        usable = dates[dates < t]
        if window != "growing":
            usable = usable[-int(window):]
        rows.append({
            "holding_month": str(t.to_period("M") + 1),
            "decision_date": t,
            "train_start": usable[0],
            "train_end": usable[-1],
            "val_start": usable[-config.VALIDATION_MONTHS],
            "n_train_months": len(usable),
            "window": str(window),
        })
    return pd.DataFrame(rows)


def rows_for(panel, step, universe):
    """Split the panel for one calendar row into train, validation and test.

    step: one row of walk_forward_calendar(). Returns three DataFrames.
    The fit slice is train minus validation; after tuning, refit on train.
    """
    p = panel[panel["universe"] == universe]
    train = p[(p["date"] >= step["train_start"]) & (p["date"] <= step["train_end"])]
    val = train[train["date"] >= step["val_start"]]
    test = p[p["date"] == step["decision_date"]]
    return train, val, test


def check_calendar(calendar, panel):
    """Checks that no training or validation row can see the test month."""
    rows = []
    ok = (calendar["train_end"] < calendar["decision_date"]).all()
    rows.append({"check": "train ends before decision", "status": "PASS" if ok else "FAIL",
                 "detail": "every training window ends at least one month-end before t"})
    first, last = calendar["holding_month"].iloc[0], calendar["holding_month"].iloc[-1]
    ok = (first == config.TEST_FIRST_HOLDING) and (last == config.TEST_LAST_HOLDING)
    rows.append({"check": "test period", "status": "PASS" if ok else "FAIL",
                 "detail": f"{len(calendar)} test months, {first} to {last}"})
    ok = calendar["holding_month"].is_unique
    rows.append({"check": "one prediction per month", "status": "PASS" if ok else "FAIL",
                 "detail": "no test month repeated"})
    labels_known = panel[panel["date"].isin(calendar["train_end"])]["next_return"].notna().all()
    rows.append({"check": "training labels known", "status": "PASS" if labels_known else "FAIL",
                 "detail": "the newest training row of every window has a label"})
    return pd.DataFrame(rows)
