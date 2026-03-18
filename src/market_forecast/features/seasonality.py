from __future__ import annotations

import numpy as np
import pandas as pd


def seasonality_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    week = index.isocalendar().week.astype(float)
    month = index.month.astype(float)
    return pd.DataFrame(
        {
            "week_sin": np.sin(2 * np.pi * week / 52.0),
            "week_cos": np.cos(2 * np.pi * week / 52.0),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
        },
        index=index,
    )
