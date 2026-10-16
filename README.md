# Return Prediction versus Rank Classification in Machine-Learning Portfolio Selection

Does a model trained to pick next month's top 3 ETFs build a better portfolio after fees than the same model trained to predict returns?

## Notebooks

| Notebook | WBS | Output |
|---|---|---|
| 1 Data Download | 2.2, 2.3 | Raw Prices, Daily Prices, Quality Report |
| 2 Features and Labels | 2.4, 2.5, 3.1 | Monthly Data, Training Windows, Variables |

## Code

| File | Purpose |
|---|---|
| `config.py` | Tickers, dates, windows, file names |
| `download.py` | Download and freeze prices |
| `cleaning.py` | Quality checks and cleaning |
| `features.py` | Features, labels, monthly table |
| `windows.py` | Training and validation windows |
| `variables.py` | Definition of every saved column |

Data lives on Google Drive in `Capstone/data`. Open a notebook in Colab and run all cells.
