from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def single_asset_figure(price: pd.Series, forecast: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name="price"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="forecast"))
    fig.update_layout(title="Single Asset Forecast", template="plotly_white")
    return fig
