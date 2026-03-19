from __future__ import annotations

import pandas as pd

from market_forecast.dashboard.multi_asset import correlation_heatmap
from market_forecast.dashboard.single_asset import single_asset_figure


def build_single_asset_dashboard(price: pd.Series, forecast: pd.Series):
    return single_asset_figure(price, forecast)


def build_multi_asset_dashboard(corr: pd.DataFrame):
    return correlation_heatmap(corr)
