"""Train a next-day SPY direction classifier with walk-forward validation and backtesting.

This module now supports running multiple experimental scenarios so you can compare
different model classes, validation windows, and decision thresholds in a single run.
Use ``--grid-search`` to execute the built-in roster of 10 experiments.
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


@dataclass
class ModelResults:
    """Container for model evaluation outputs."""

    accuracy: float
    precision: float
    recall: float
    roc_auc: float


@dataclass
class BacktestResults:
    """Container for backtest metrics."""

    cumulative_return: float
    buy_and_hold_return: float
    daily_hit_rate: float
    active_hit_rate: float
    trade_count: int
    win_rate: float
    avg_trade_return: float
    avg_holding_period: float
    turnover: float
    total_transaction_cost: float


@dataclass
class ExperimentReport:
    """Summary of metrics for a single training/backtest experiment."""

    name: str
    model: str
    test_size: float
    threshold: float
    cv_details: Sequence[ModelResults]
    cv_metrics: Dict[str, float]
    holdout: ModelResults
    backtest: BacktestResults

    def to_row(self) -> List[object]:
        """Represent the experiment as a tabular row."""

        return [
            self.name,
            self.model,
            f"{self.test_size:.2f}",
            f"{self.threshold:.2f}",
            f"{self.cv_metrics['accuracy_mean']:.3f}",
            f"{self.cv_metrics['roc_auc_mean']:.3f}",
            f"{self.holdout.accuracy:.3f}",
            f"{self.holdout.roc_auc:.3f}",
            f"{self.backtest.cumulative_return:.2%}",
            f"{self.backtest.buy_and_hold_return:.2%}",
            f"{self.backtest.daily_hit_rate:.2%}",
            f"{self.backtest.active_hit_rate:.2%}",
            f"{self.backtest.trade_count}",
            f"{self.backtest.win_rate:.2%}",
        ]


@dataclass
class ScenarioConfig:
    """Configuration for a single experimental run."""

    name: str
    model: str
    test_size: float
    cv_splits: int
    threshold: float
    optimize_threshold: bool = False
    threshold_grid: Sequence[float] = (0.45, 0.5, 0.55, 0.6, 0.65)
    retrain_frequency: int = 0
    retrain_window: Optional[int] = None


def download_price_history(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance. Check the symbol and date range.")
    df.index = df.index.tz_localize(None)
    df = df.rename(columns=str.lower)
    return df


def load_price_history_from_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV data from a CSV exported by Yahoo Finance."""

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.rename(columns=str.lower)
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index for a price series."""

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_macro_context(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download auxiliary macro/regime indicators aligned by date."""

    symbols = {
        "^VIX": "vix_close",
        "^TNX": "ust10y_yield",
        "^IRX": "ust3m_yield",
    }
    frames: List[pd.Series] = []
    for symbol, column_name in symbols.items():
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty:
            continue
        series = data.get("Adj Close") if "Adj Close" in data else data.get("Close")
        if series is None:
            continue
        series = series.rename(column_name)
        frames.append(series)

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1).sort_index()
    macro.index = macro.index.tz_localize(None)
    return macro.ffill()


def engineer_features(
    df: pd.DataFrame, macro: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Create technical, regime, and macro features for the ML model."""

    features = df.copy()
    features["return_1d"] = features["close"].pct_change()
    features["return_2d"] = features["close"].pct_change(2)
    features["return_5d"] = features["close"].pct_change(5)
    features["return_10d"] = features["close"].pct_change(10)
    features["return_20d"] = features["close"].pct_change(20)
    features["return_60d"] = features["close"].pct_change(60)
    features["volatility_5d"] = features["return_1d"].rolling(5).std()
    features["volatility_10d"] = features["return_1d"].rolling(10).std()
    features["volatility_20d"] = features["return_1d"].rolling(20).std()
    features["volatility_60d"] = features["return_1d"].rolling(60).std()
    features["ma_ratio_5"] = features["close"] / features["close"].rolling(5).mean()
    features["ma_ratio_10"] = features["close"] / features["close"].rolling(10).mean()
    features["ma_ratio_20"] = features["close"] / features["close"].rolling(20).mean()
    features["ma_ratio_50"] = features["close"] / features["close"].rolling(50).mean()
    features["ma_ratio_100"] = features["close"] / features["close"].rolling(100).mean()
    features["ma_ratio_200"] = features["close"] / features["close"].rolling(200).mean()
    features["high_low_pct"] = (features["high"] - features["low"]) / features["close"]
    features["close_open_pct"] = (features["close"] - features["open"]) / features["open"]
    features["volume_change"] = features["volume"].pct_change()
    volume_mean_20 = features["volume"].rolling(20).mean()
    volume_std_20 = features["volume"].rolling(20).std()
    features["volume_zscore_20"] = (features["volume"] - volume_mean_20) / (volume_std_20 + 1e-9)
    features["volume_percentile_60"] = (
        features["volume"].rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    )
    high_roll_max = features["high"].rolling(20).max()
    low_roll_min = features["low"].rolling(20).min()
    features["range_breakout_20"] = (features["close"] - low_roll_min) / (high_roll_max - low_roll_min + 1e-9)
    features["rsi_14"] = compute_rsi(features["close"], 14)
    features["rsi_3"] = compute_rsi(features["close"], 3)
    features["drawdown_20d"] = features["close"] / features["close"].rolling(20).max() - 1
    vol_mean_252 = features["volatility_20d"].rolling(252, min_periods=60).mean()
    vol_std_252 = features["volatility_20d"].rolling(252, min_periods=60).std()
    features["volatility_20d_zscore"] = (
        features["volatility_20d"] - vol_mean_252
    ) / (vol_std_252 + 1e-9)
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month

    if macro is not None and not macro.empty:
        macro_aligned = macro.reindex(features.index).ffill()
        features = features.join(macro_aligned)
        if "vix_close" in features:
            features["vix_change_5d"] = features["vix_close"].pct_change(5)
            features["vix_change_1d"] = features["vix_close"].pct_change()
            features["vix_return_corr_20"] = (
                features["return_1d"].rolling(20).corr(features["vix_close"].pct_change())
            )
        if {"ust10y_yield", "ust3m_yield"}.issubset(features.columns):
            features["yield_curve_slope"] = features["ust10y_yield"] - features["ust3m_yield"]
            features["yield_curve_change_20d"] = features["yield_curve_slope"].diff(20)
            features["ust10y_change_5d"] = features["ust10y_yield"].diff(5)

    features["target"] = (features["close"].shift(-1) > features["close"]).astype(int)
    features["next_return"] = features["close"].shift(-1) / features["close"] - 1
    features = features.dropna()
    return features


def build_pipeline(model_type: str) -> Pipeline:
    """Create the sklearn pipeline used for training and inference."""

    model_type = model_type.lower()
    if model_type == "logistic":
        classifier = LogisticRegression(max_iter=10_000, class_weight="balanced", C=0.5)
        steps = [("scaler", StandardScaler()), ("classifier", classifier)]
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )
        steps = [("classifier", classifier)]
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=400,
            max_depth=3,
            random_state=42,
        )
        steps = [("classifier", classifier)]
    else:
        raise ValueError(
            "Unsupported model type. Choose from 'logistic', 'random_forest', or 'gradient_boosting'."
        )
    return Pipeline(steps=steps)


def time_series_cross_validation(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int
) -> List[ModelResults]:
    """Evaluate the pipeline using expanding-window walk-forward validation."""

    cv = TimeSeriesSplit(n_splits=n_splits)
    metrics: List[ModelResults] = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = clone(pipeline)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics.append(
            ModelResults(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_test, y_proba),
            )
        )
    return metrics


def format_cv_table(metrics: Iterable[ModelResults]) -> str:
    """Create a human-readable table of cross-validation metrics."""

    rows = [
        [idx + 1, m.accuracy, m.precision, m.recall, m.roc_auc]
        for idx, m in enumerate(metrics)
    ]
    headers = ["Fold", "Accuracy", "Precision", "Recall", "ROC AUC"]
    return tabulate(rows, headers=headers, floatfmt=".3f")


def summarize_cv_metrics(metrics: Sequence[ModelResults]) -> Dict[str, float]:
    """Compute aggregate statistics from cross-validation folds."""

    accuracy = [m.accuracy for m in metrics]
    roc_auc = [m.roc_auc for m in metrics]
    precision = [m.precision for m in metrics]
    recall = [m.recall for m in metrics]
    return {
        "accuracy_mean": float(pd.Series(accuracy).mean()),
        "accuracy_std": float(pd.Series(accuracy).std(ddof=0)),
        "roc_auc_mean": float(pd.Series(roc_auc).mean()),
        "roc_auc_std": float(pd.Series(roc_auc).std(ddof=0)),
        "precision_mean": float(pd.Series(precision).mean()),
        "recall_mean": float(pd.Series(recall).mean()),
    }


def holdout_predictions(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    retrain_frequency: int = 0,
    retrain_window: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, Optional[Pipeline]]:
    """Generate hold-out probabilities with optional walk-forward retraining."""

    if retrain_frequency <= 0:
        model = clone(pipeline)
        X_train, y_train = train_df[predictors], train_df["target"]
        X_test = test_df[predictors]
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        return (
            pd.Series(y_proba, index=X_test.index, name="probability"),
            pd.Series(y_pred, index=X_test.index, name="prediction"),
            model,
        )

    history_X = train_df[predictors].copy()
    history_y = train_df["target"].copy()
    probabilities: List[float] = []
    predictions: List[int] = []
    model: Optional[Pipeline] = None

    for idx, (timestamp, row) in enumerate(test_df.iterrows()):
        if model is None or idx % retrain_frequency == 0:
            if retrain_window is not None:
                X_fit = history_X.iloc[-retrain_window:]
                y_fit = history_y.iloc[-retrain_window:]
            else:
                X_fit = history_X
                y_fit = history_y
            model = clone(pipeline)
            model.fit(X_fit, y_fit)

        row_X = row[predictors].to_frame().T
        proba = float(model.predict_proba(row_X)[0, 1])
        pred = int(proba >= 0.5)
        probabilities.append(proba)
        predictions.append(pred)

        history_X = pd.concat([history_X, row_X])
        history_y = pd.concat([history_y, pd.Series(row["target"], index=[timestamp])])

    proba_series = pd.Series(probabilities, index=test_df.index, name="probability")
    pred_series = pd.Series(predictions, index=test_df.index, name="prediction")
    return proba_series, pred_series, model


def backtest_strategy(
    signals: pd.Series,
    next_returns: pd.Series,
    transaction_cost_bps: float,
) -> BacktestResults:
    """Run a long-only backtest with transaction costs and trade diagnostics."""

    aligned_returns = next_returns.loc[signals.index]
    cost_rate = transaction_cost_bps / 10_000.0
    position_change = signals.diff().abs().fillna(signals.abs())
    costs = position_change * cost_rate
    gross_returns = signals * aligned_returns
    net_returns = gross_returns - costs

    trade_net_returns = net_returns.copy()
    changes = position_change.to_numpy()
    signal_values = signals.to_numpy()
    for idx, (signal, change) in enumerate(zip(signal_values, changes)):
        if signal == 0 and change > 0 and idx > 0:
            trade_net_returns.iloc[idx - 1] += net_returns.iloc[idx]
            trade_net_returns.iloc[idx] = 0.0

    cumulative_return = float((1 + net_returns).prod() - 1)
    buy_and_hold_return = float((1 + aligned_returns).prod() - 1)

    daily_hits = (signals == (aligned_returns > 0).astype(int)).astype(int)
    daily_hit_rate = float(daily_hits.mean()) if len(daily_hits) else 0.0

    active_mask = signals > 0
    if active_mask.any():
        active_hits = float((aligned_returns[active_mask] > 0).mean())
    else:
        active_hits = 0.0

    trade_returns: List[float] = []
    holding_periods: List[int] = []
    running_return = 1.0
    holding_days = 0
    trade_values = trade_net_returns.to_numpy()

    for idx, signal in enumerate(signal_values):
        if signal == 1:
            if holding_days == 0:
                running_return = 1.0
            running_return *= 1 + trade_values[idx]
            holding_days += 1
            next_signal = signal_values[idx + 1] if idx + 1 < len(signal_values) else 0
            if next_signal == 0:
                trade_returns.append(running_return - 1)
                holding_periods.append(holding_days)
                holding_days = 0
        else:
            holding_days = 0

    trade_count = len(trade_returns)
    if trade_count:
        win_rate = float(np.mean([ret > 0 for ret in trade_returns]))
        avg_trade_return = float(np.mean(trade_returns))
        avg_holding_period = float(np.mean(holding_periods))
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        avg_holding_period = 0.0

    turnover = float(position_change.sum())
    total_cost = float(costs.sum())

    return BacktestResults(
        cumulative_return=cumulative_return,
        buy_and_hold_return=buy_and_hold_return,
        daily_hit_rate=daily_hit_rate,
        active_hit_rate=active_hits,
        trade_count=trade_count,
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        avg_holding_period=avg_holding_period,
        turnover=turnover,
        total_transaction_cost=total_cost,
    )


def optimize_probability_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    thresholds: Sequence[float],
    validation_fraction: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
    """Search for the best classification threshold on a validation slice."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    split_idx = int(len(X_train) * (1 - validation_fraction))
    if split_idx <= 0 or split_idx >= len(X_train):
        raise ValueError("Not enough observations to create a validation split")

    X_subtrain, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_subtrain, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    model = clone(pipeline)
    model.fit(X_subtrain, y_subtrain)
    val_proba = model.predict_proba(X_val)[:, 1]

    best_threshold = thresholds[0]
    best_score = -float("inf")
    for threshold in thresholds:
        preds = (val_proba >= threshold).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, {"validation_f1": float(best_score), "validation_size": len(y_val)}


def split_train_test(features: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split features into chronological train/test sets."""

    split_idx = int(len(features) * (1 - test_size))
    train = features.iloc[:split_idx]
    test = features.iloc[split_idx:]
    return train, test


def run_experiment(
    scenario: ScenarioConfig,
    predictors: Sequence[str],
    features: pd.DataFrame,
    transaction_cost_bps: float,
) -> ExperimentReport:
    """Execute training, validation, and backtesting for a single scenario."""

    train, test = split_train_test(features, test_size=scenario.test_size)
    X_train, y_train = train[predictors], train["target"]
    X_test, y_test = test[predictors], test["target"]

    pipeline = build_pipeline(scenario.model)

    cv_metrics = time_series_cross_validation(pipeline, X_train, y_train, n_splits=scenario.cv_splits)
    cv_summary = summarize_cv_metrics(cv_metrics)

    threshold = scenario.threshold
    threshold_info: Dict[str, float] = {}
    if scenario.optimize_threshold:
        threshold, threshold_info = optimize_probability_threshold(
            pipeline, X_train, y_train, thresholds=scenario.threshold_grid
        )

    probas, preds, _model = holdout_predictions(
        pipeline,
        train,
        test,
        predictors,
        retrain_frequency=scenario.retrain_frequency,
        retrain_window=scenario.retrain_window,
    )
    holdout = ModelResults(
        accuracy=accuracy_score(y_test, preds),
        precision=precision_score(y_test, preds, zero_division=0),
        recall=recall_score(y_test, preds, zero_division=0),
        roc_auc=roc_auc_score(y_test, probas),
    )
    signals = (probas >= threshold).astype(int)
    backtest = backtest_strategy(signals, test["next_return"], transaction_cost_bps)

    if threshold_info:
        print(
            f"Threshold optimized to {threshold:.2f} on validation F1={threshold_info['validation_f1']:.3f} "
            f"(validation size={int(threshold_info['validation_size'])})"
        )

    return ExperimentReport(
        name=scenario.name,
        model=scenario.model,
        test_size=scenario.test_size,
        threshold=threshold,
        cv_details=cv_metrics,
        cv_metrics=cv_summary,
        holdout=holdout,
        backtest=backtest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to download from Yahoo Finance")
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date for historical data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=dt.date.today().isoformat(),
        help="End date for historical data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Optional path to a CSV file with historical OHLCV data",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of observations to reserve for the hold-out evaluation",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of folds for walk-forward cross-validation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Probability threshold for entering a long position during backtests",
    )
    parser.add_argument(
        "--model",
        default="logistic",
        choices=["logistic", "random_forest", "gradient_boosting"],
        help="Model family to train when not running grid search",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run the built-in 10-scenario sweep and report a comparison table",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Calibrate the decision threshold on a validation slice of the training data",
    )
    parser.add_argument(
        "--retrain-frequency",
        type=int,
        default=0,
        help="Number of hold-out observations between walk-forward retraining steps (0 disables)",
    )
    parser.add_argument(
        "--retrain-window",
        type=int,
        help="Optional number of most recent observations to keep when retraining (expanding if omitted)",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="Transaction cost in basis points charged each time the position changes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = dt.datetime.fromisoformat(args.start).date()
    end_date = dt.datetime.fromisoformat(args.end).date()

    if args.csv_path:
        print(f"Loading data from {args.csv_path}...")
        raw_prices = load_price_history_from_csv(args.csv_path)
        raw_prices = raw_prices.loc[start_date:end_date]
    else:
        print(f"Downloading data for {args.symbol} from {start_date} to {end_date}...")
        raw_prices = download_price_history(args.symbol, start_date, end_date)

    if raw_prices.empty:
        raise ValueError("No price history available for the requested window.")

    print(f"Retrieved {len(raw_prices)} rows of data.")

    context_start = max(dt.date(1900, 1, 1), start_date - dt.timedelta(days=400))
    macro_features = pd.DataFrame()
    try:
        print("Downloading macro and regime context...")
        macro_features = download_macro_context(context_start, end_date)
        if macro_features.empty:
            print("Warning: macro context download returned no data; continuing without it.")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Warning: failed to download macro context ({exc}); continuing without it.")

    print("Engineering features...")
    features = engineer_features(raw_prices, macro_features)
    predictors = [
        "return_1d",
        "return_2d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "volatility_60d",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_20",
        "ma_ratio_50",
        "ma_ratio_100",
        "ma_ratio_200",
        "high_low_pct",
        "close_open_pct",
        "volume_change",
        "volume_zscore_20",
        "volume_percentile_60",
        "range_breakout_20",
        "drawdown_20d",
        "volatility_20d_zscore",
        "rsi_3",
        "rsi_14",
        "day_of_week",
        "month",
        "vix_close",
        "vix_change_1d",
        "vix_change_5d",
        "vix_return_corr_20",
        "ust10y_yield",
        "ust10y_change_5d",
        "ust3m_yield",
        "yield_curve_slope",
        "yield_curve_change_20d",
    ]
    predictors = [col for col in predictors if col in features.columns]

    def build_default_scenarios() -> List[ScenarioConfig]:
        base_splits = max(args.cv_splits, 3)
        return [
            ScenarioConfig(
                name="LogReg-Std",
                model="logistic",
                test_size=args.test_size,
                cv_splits=base_splits,
                threshold=0.50,
                optimize_threshold=False,
            ),
            ScenarioConfig(
                name="LogReg-OptThresh",
                model="logistic",
                test_size=args.test_size,
                cv_splits=base_splits,
                threshold=args.threshold,
                optimize_threshold=True,
            ),
            ScenarioConfig(
                name="LogReg-LongHold",
                model="logistic",
                test_size=0.30,
                cv_splits=base_splits,
                threshold=0.55,
                optimize_threshold=True,
                retrain_frequency=20,
            ),
            ScenarioConfig(
                name="LogReg-DeepCV",
                model="logistic",
                test_size=0.25,
                cv_splits=8,
                threshold=0.55,
                optimize_threshold=False,
            ),
            ScenarioConfig(
                name="RF-Balanced",
                model="random_forest",
                test_size=args.test_size,
                cv_splits=base_splits,
                threshold=0.50,
                optimize_threshold=False,
            ),
            ScenarioConfig(
                name="RF-OptThresh",
                model="random_forest",
                test_size=0.25,
                cv_splits=base_splits,
                threshold=0.55,
                optimize_threshold=True,
                retrain_frequency=20,
                retrain_window=504,
            ),
            ScenarioConfig(
                name="RF-WideSplit",
                model="random_forest",
                test_size=0.35,
                cv_splits=6,
                threshold=0.55,
                optimize_threshold=False,
            ),
            ScenarioConfig(
                name="GBM-Std",
                model="gradient_boosting",
                test_size=args.test_size,
                cv_splits=base_splits,
                threshold=0.50,
                optimize_threshold=False,
            ),
            ScenarioConfig(
                name="GBM-OptThresh",
                model="gradient_boosting",
                test_size=0.25,
                cv_splits=base_splits,
                threshold=0.55,
                optimize_threshold=True,
                retrain_frequency=20,
                retrain_window=504,
            ),
            ScenarioConfig(
                name="GBM-DeepCV",
                model="gradient_boosting",
                test_size=0.30,
                cv_splits=7,
                threshold=0.60,
                optimize_threshold=False,
            ),
        ]

    if args.grid_search:
        print("Running 10-scenario sweep...")
        reports: List[ExperimentReport] = []
        for scenario in build_default_scenarios():
            print(f"\n=== Scenario: {scenario.name} ({scenario.model}) ===")
            report = run_experiment(
                scenario, predictors, features, transaction_cost_bps=args.transaction_cost_bps
            )
            print(
                f"Hold-out Accuracy={report.holdout.accuracy:.3f}, ROC AUC={report.holdout.roc_auc:.3f}, "
                f"Strategy Return={report.backtest.cumulative_return:.2%}, "
                f"Buy&Hold={report.backtest.buy_and_hold_return:.2%}, Daily Hit={report.backtest.daily_hit_rate:.2%}, "
                f"Active Hit={report.backtest.active_hit_rate:.2%}, Trades={report.backtest.trade_count}, "
                f"Win Rate={report.backtest.win_rate:.2%}, Costs={report.backtest.total_transaction_cost:.2%}"
            )
            reports.append(report)

        headers = [
            "Scenario",
            "Model",
            "Test Size",
            "Threshold",
            "CV Acc (mean)",
            "CV ROC AUC (mean)",
            "Hold-out Acc",
            "Hold-out ROC AUC",
            "Strategy Return",
            "Buy&Hold Return",
            "Daily Hit Rate",
            "Active Hit Rate",
            "Trades",
            "Win Rate",
        ]
        table = [report.to_row() for report in reports]
        print("\n=== Scenario comparison ===")
        print(tabulate(table, headers=headers, tablefmt="github"))

        best = max(reports, key=lambda r: r.backtest.cumulative_return)
        print(
            "\nBest scenario by strategy return: "
            f"{best.name} ({best.model}) with {best.backtest.cumulative_return:.2%} cumulative return"
        )
        return

    scenario = ScenarioConfig(
        name="Custom",
        model=args.model,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        retrain_frequency=args.retrain_frequency,
        retrain_window=args.retrain_window,
    )
    report = run_experiment(
        scenario, predictors, features, transaction_cost_bps=args.transaction_cost_bps
    )

    print("Performing walk-forward cross-validation...")
    print(format_cv_table(report.cv_details))

    print("\nTraining on the full training window and evaluating the hold-out period...")
    print(
        f"Hold-out metrics -> Accuracy: {report.holdout.accuracy:.3f}, "
        f"Precision: {report.holdout.precision:.3f}, Recall: {report.holdout.recall:.3f}, "
        f"ROC AUC: {report.holdout.roc_auc:.3f}"
    )

    print("\nRunning backtest on the hold-out window...")
    print(
        f"Strategy cumulative return: {report.backtest.cumulative_return:.2%}\n"
        f"Buy & hold return: {report.backtest.buy_and_hold_return:.2%}\n"
        f"Daily hit rate: {report.backtest.daily_hit_rate:.2%} (active days: {report.backtest.active_hit_rate:.2%})\n"
        f"Trades: {report.backtest.trade_count} | Win rate: {report.backtest.win_rate:.2%}\n"
        f"Avg trade return: {report.backtest.avg_trade_return:.2%} | Avg holding period: {report.backtest.avg_holding_period:.1f} days\n"
        f"Turnover: {report.backtest.turnover:.2f} | Total transaction cost: {report.backtest.total_transaction_cost:.2%}"
    )


if __name__ == "__main__":
    main()
