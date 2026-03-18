from __future__ import annotations

import pandas as pd
import plotly.express as px


def correlation_heatmap(corr: pd.DataFrame):
    return px.imshow(corr, text_auto=True, title="EWMA Correlation Matrix")
