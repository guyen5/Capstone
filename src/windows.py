"""Training windows for the walk-forward test (WBS 3.1).

At each decision date t, the model is refit on rows whose labels are already
known (rows dated before t) and predicts the coming month. The last 12
months of each window are the validation slice for tuning; the model is then
refit on the whole window. "growing" uses every month since 2010 (primary);
36 and 60 use only the most recent 3 or 5 years (robustness).
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
