from __future__ import annotations

import pandas as pd

from market_forecast.backtest.engine import rolling_naive_backtest


def baseline_research_backtest(returns: pd.Series) -> dict[str, float]:
    result = rolling_naive_backtest(returns, train_window=252, test_window=20, step=20)
    return {
        "rmse": result.rmse,
        "directional_accuracy": result.directional_accuracy,
    }
