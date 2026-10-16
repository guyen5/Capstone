# Return Prediction versus Rank Classification in Machine-Learning Portfolio Selection

**A walk-forward test of the decision-induced ranking hypothesis on ETF portfolios**

Capstone project, Minerva University, 2026-2027. Advisor: Prof. Engin Volkan.

## The question

Wang and Hasuike (2026, arXiv 2605.01176) show that a mean-variance optimizer
with transaction costs picks assets by the *order* of its inputs, not their
size, and that models trained to predict returns end up replacing over 80% of
the portfolio every month. This project tests the direct implication: does a
model trained to **classify which ETFs will be in next month's top 3** build
better portfolios after fees than the same model trained to **predict next
month's return**?

- **H1:** with the same inputs, model and portfolio rule, the classification-trained model earns a higher average monthly return after a 0.5% fee.
- **H0:** no difference.
- **Test:** 8 ETFs, monthly walk-forward backtest, January 2020 to September 2026, block and Bayesian bootstrap.

## Repository map

Files are numbered by the work package (WBS) they deliver. The WBS itself is
the Google Sheet "Capstone WBS".

```
Capstone/
├── README.md                              this page
├── requirements.txt                       libraries Colab installs
├── notebooks/                             run these in Colab, in order
│   ├── 2.2_download_and_quality.ipynb       WBS 2.2-2.3  download, freeze, check, clean
│   └── 2.4_features_and_panel.ipynb         WBS 2.4-2.5, 3.1  features, labels, panel, calendar
├── src/                                   code the notebooks import
│   ├── config.py                            every setting in one place
│   ├── download.py                          WBS 2.2  Yahoo download, freeze, load
│   ├── quality.py                           WBS 2.3  quality checks and cleaning
│   ├── features.py                          WBS 2.4  indicators, labels, monthly panel
│   ├── dictionary.py                        WBS 2.5  definition of every saved column
│   └── walk_forward.py                      WBS 3.1  train / validation / test windows
└── tests/
    └── test_pipeline.py                     checks the pipeline on simulated prices
```

Coming next, in WBS order: `3.2_backtest_engine`, `4_models`,
`5_hypothesis_test`, `6_robustness`.

## Where things live

| What | Where |
|---|---|
| Code and notebooks | This repository |
| Data | Google Drive, `MyDrive/Capstone/data/` (`raw/`, `processed/`, `reports/`) |
| Plan and progress | Google Sheet "Capstone WBS" |
| Data definitions | Google Doc "Capstone Data Specification" |

Data never goes into this repository (see `.gitignore`).

## How to run

1. In Colab: File, Open notebook, GitHub tab, `guyen5/Capstone`, pick a notebook.
2. Runtime, Run all. The first cell mounts Drive, pulls this repository and installs the libraries.
3. File, Save a copy in GitHub, same path, to keep the outputs.

Each notebook reads the previous one's output from Drive, so they run in order
once and never need to be rerun unless the code changes.

## Current data (freeze of 2026-10-16)

| Table | Rows | Status |
|---|---|---|
| `raw/prices_raw_20261016.csv` | 71,408 (16 ETFs x 4,463 days) | frozen, MD5 80113bee... |
| `processed/prices_clean_v1.parquet` | 71,408 | 0 FAIL, 1 WARN (JPXN zero volume) |
| `processed/monthly_panel_v1.parquet` | 3,618 (ETF_A 1,608, ETF_B 2,010) | 29 of 29 checks pass |
| `processed/walk_forward_calendar_v1.parquet` | 243 (81 test months x 3 windows) | all checks pass |
