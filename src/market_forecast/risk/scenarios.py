from __future__ import annotations

import pandas as pd


def stress_scenario(returns: pd.DataFrame, shock_sigma: float = 2.0) -> pd.Series:
    mu = returns.mean()
    sd = returns.std(ddof=0)
    return mu - shock_sigma * sd
