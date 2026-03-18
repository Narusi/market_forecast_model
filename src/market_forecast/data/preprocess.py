from __future__ import annotations

import pandas as pd


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("W-FRI").last().dropna(how="all")


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
