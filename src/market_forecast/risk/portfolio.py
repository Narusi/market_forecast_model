from __future__ import annotations

import pandas as pd


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    w = 1.0 / returns.shape[1]
    return returns.mean(axis=1) * w * returns.shape[1]
