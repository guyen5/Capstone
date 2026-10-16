# Predict the Return or Predict the Rank?

Capstone project, Guyen Tumurbat, Minerva University. Advisor: Prof. Engin Volkan.

Tests whether an ML model trained to predict which ETFs will finish in next
month's top 3 builds better portfolios after fees than the same model trained
to predict next month's returns. Builds on Wang and Hasuike (2026), arXiv 2605.01176.

## Where things live

| What | Where |
|---|---|
| Code (`src/`) and notebooks (repo root) | This repository |
| Data (raw downloads, clean tables, reports) | Google Drive, `MyDrive/Capstone/data/` |
| Running the notebooks | Google Colab |

Data is never committed here (see `.gitignore`).

## Reproduce

1. Open `01_data_pull_and_freeze.ipynb` in Colab (File, Open notebook, GitHub tab, `guyen5/Capstone`).
2. Run all cells. The first cell mounts Drive, clones this repository and installs `requirements.txt`.
3. Later notebooks start from `data/processed/prices_clean_v1.parquet`.

## Notebooks

| Notebook | Work package | Output |
|---|---|---|
| 01_data_pull_and_freeze | 2.2, 2.3 | Frozen raw prices, quality report, `prices_clean_v1` |
| 02_features_and_panel | 2.4, 2.5 | `monthly_panel_v1` (next) |
