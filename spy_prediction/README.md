# Project Overview

In this project, we build a machine learning pipeline that forecasts whether the SPDR S&P 500 ETF Trust (SPY) will close higher or lower on the following trading day. The workflow covers data ingestion, feature engineering, model training with walk-forward validation, and a simple backtest that compares the model-driven strategy to a passive buy-and-hold benchmark.

## Project Steps

* Download daily SPY data with the `yfinance` package.
* Engineer technical, regime, and macro indicators (including VIX levels and Treasury yields) that describe the recent market context.
* Train and tune classification models (logistic regression, random forest, gradient boosting) using time-series aware cross-validation.
* Evaluate the model on a hold-out set, calibrate the trading threshold, and run a vectorized backtest with walk-forward retraining, transaction costs, and trade-level diagnostics to measure performance.

## Code

* `train_spy_model.py` – end-to-end script that orchestrates data downloading, preprocessing, model training, evaluation, and backtesting.

## Local Setup

### Installation

Install the project dependencies in your Python 3.9+ environment:

```bash
pip install -r requirements.txt
```

If you would rather install packages individually, make sure to add:

* pandas
* numpy
* scikit-learn
* yfinance
* tabulate

### Usage

Run the full pipeline from the repository root with:

```bash
python -m spy_prediction.train_spy_model
```

By default the script pulls data starting in 2010, but you can customize the training window, model family, probability threshold, walk-forward retraining cadence, and transaction cost assumptions. Use `--help` to see available options.

To automatically compare ten preconfigured scenarios (mixing models, validation splits, and threshold strategies) and pick the most robust backtest, add `--grid-search`:

```bash
python -m spy_prediction.train_spy_model --grid-search
```

For a single custom experiment with automatic threshold calibration, combine `--model`, `--test-size`, and `--optimize-threshold`:

```bash
python -m spy_prediction.train_spy_model --model random_forest --test-size 0.3 --optimize-threshold
```

To activate walk-forward retraining and include round-trip costs, use the new options:

```bash
python -m spy_prediction.train_spy_model \
    --model gradient_boosting \
    --retrain-frequency 20 \
    --retrain-window 504 \
    --transaction-cost-bps 5
```

This example refreshes the model every 20 hold-out observations using the most recent 504 training samples (approximately two years) and subtracts 5 bps each time the strategy enters or exits a position.

If you do not have internet access or want to reuse a previously downloaded Yahoo! Finance export, provide the CSV path explicitly:

```bash
python -m spy_prediction.train_spy_model --csv-path path/to/spy.csv
```

> **Note:** The script attempts to download auxiliary macro indicators (VIX and Treasury yields) to build the regime-aware features. If the download fails (e.g., offline environments), the warnings can be ignored and the pipeline will fall back to the purely technical feature set.

### Data

All data is downloaded on demand from Yahoo! Finance through the `yfinance` API, so no local files are required.
