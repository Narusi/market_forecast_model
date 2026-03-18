from __future__ import annotations

import pandas as pd


def dynamic_thresholds(predictions: pd.Series, z: float = 0.5) -> tuple[float, float]:
    mu = float(predictions.mean())
    sd = float(predictions.std(ddof=0))
    return mu + z * sd, mu - z * sd
