# Return Prediction versus Rank Classification in Machine-Learning Portfolio Selection

**A walk-forward test of the decision-induced ranking hypothesis on ETF portfolios**

Capstone project, Guyen Tumurbat, Minerva University. Advisor: Prof. Engin Volkan.

## Overview

Portfolio models built on machine-learning forecasts usually train the model to
predict returns as accurately as possible. Wang and Hasuike (2026, arXiv
2605.01176) show that a mean-variance optimizer with transaction costs selects
assets by the ordering of adjusted scores rather than their magnitudes, and that
return-trained models therefore inflate their predictions and replace more than
80% of the portfolio each month. This project tests the implication directly:
does training the same model to classify which ETFs will rank in next month's
top three, instead of regressing next month's return, produce better portfolios
after transaction costs?

Gradient-boosted and linear models are trained both ways in a monthly
walk-forward backtest on eight ETFs (January 2020 to September 2026, 0.5% fee),
and the difference in net returns is assessed with block and Bayesian bootstrap
inference.

**H1.** With identical inputs, model class and portfolio rule, the
classification-trained model earns a higher mean monthly net-of-cost return
than the regression-trained model. **H0.** No difference.

## Where things live

| What | Where |
|---|---|
| Code (`src/`) and notebooks (repo root) | This repository |
| Data (raw downloads, clean tables, reports) | Google Drive, `MyDrive/Capstone/data/` |
| Running the notebooks | Google Colab |

Data is never committed here (see `.gitignore`).

## Reproduce

1. Open a notebook in Colab: File, Open notebook, GitHub tab, `guyen5/Capstone`.
2. Run all cells. The first cell mounts Drive, clones this repository and installs `requirements.txt`.
3. Run the notebooks in order; each starts from the previous one's output on Drive.

## Notebooks

| Notebook | WBS | Input | Output |
|---|---|---|---|
| 01_data_pull_and_freeze | 2.2, 2.3 | Yahoo Finance | `raw/prices_raw_YYYYMMDD.csv`, `processed/prices_clean_v1.parquet`, quality report |
| 02_features_and_panel | 2.4, 2.5, 3.1 | `prices_clean_v1` | `processed/monthly_panel_v1.parquet`, `walk_forward_calendar_v1`, `data_dictionary_v1.csv`, exploration tables |
| 03_backtest_engine | 3.2 to 3.4 | panel, calendar | next |

## Code

| Module | Purpose |
|---|---|
| `src/config.py` | Every setting: tickers, dates, windows, thresholds, folders |
| `src/data_pull.py` | Download, freeze with checksum, load |
| `src/quality.py` | Quality checks and cleaning of daily prices |
| `src/features.py` | Seven indicators, month-end panel, labels, panel validation |
| `src/splits.py` | Walk-forward calendar and train / validation / test rows |
| `src/dictionary.py` | Data dictionary; every saved table is checked against it |
