from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_forecast.backtest.metrics import directional_accuracy, rmse
from market_forecast.backtest.walk_forward import walk_forward_splits


@dataclass
class BacktestResult:
    predictions: pd.Series
    truth: pd.Series
    rmse: float
    directional_accuracy: float


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> BacktestResult:
    return BacktestResult(
        predictions=y_pred,
        truth=y_true,
        rmse=rmse(y_true, y_pred),
        directional_accuracy=directional_accuracy(y_true, y_pred),
    )


def rolling_naive_backtest(returns: pd.Series, train_window: int, test_window: int, step: int) -> BacktestResult:
    preds = []
    truth = []
    for s in walk_forward_splits(returns.index, train_window, test_window, step):
        train = returns.loc[s.train_idx].dropna()
        test = returns.loc[s.test_idx].dropna()
        if train.empty or test.empty:
            continue
        pred = pd.Series(float(train.mean()), index=test.index)
        preds.append(pred)
        truth.append(test)
    y_pred = pd.concat(preds) if preds else pd.Series(dtype=float)
    y_true = pd.concat(truth) if truth else pd.Series(dtype=float)
    return evaluate_predictions(y_true, y_pred)
