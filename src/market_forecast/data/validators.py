from __future__ import annotations

import pandas as pd

from market_forecast.utils.time import enforce_datetime_index


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = enforce_datetime_index(prices)
    if prices.isna().all().all():
        raise ValueError("Price data contains only NaN values")
    return prices.ffill().dropna(how="all")
