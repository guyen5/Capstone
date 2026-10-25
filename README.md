# Reliability-Weighted Forecasts in Predict-then-Optimize Portfolios

Does weighting each return forecast by its estimated reliability, through Black-Litterman inside the same Markowitz model, give a higher after-fee Sharpe ratio at every level of turnover than plain predict-then-optimize? The project first replicates the PTO results of Wang and Hasuike (2026) on their ETF universe, then makes that one change.

## Notebooks

| Notebook | WBS | Output |
|---|---|---|
| 1 Data Download | 2.2, 2.3 | Raw Prices, Daily Prices, Quality Report |
| 2 Features and Labels | 2.4, 2.5 | Monthly Data, Training Windows (monthly), Variables |
| 3 Backtest Engine | 3.2, 3.3 | Daily Features, Daily Training Windows, Covariances |

## Code

| File | Purpose |
|---|---|
| `config.py` | Tickers, dates, windows, experiment settings, file names |
| `download.py` | Download and freeze prices |
| `cleaning.py` | Quality checks and cleaning |
| `features.py` | Features and labels, monthly and daily tables |
| `windows.py` | Training and validation windows, monthly and daily |
| `covariance.py` | Sample and Ledoit-Wolf covariance at each decision date |
| `variables.py` | Definition of every saved column |

Data lives on Google Drive in `Capstone/data`. Open a notebook in Colab and run all cells. The tests run on simulated prices with `python -m pytest tests`.
