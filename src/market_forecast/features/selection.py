from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 20:
        return 0.0
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def select_predictive_features(
    features: pd.DataFrame,
    target: pd.Series,
    min_abs_ic: float = 0.01,
) -> list[str]:
    selected = []
    for col in features.columns:
        ic = information_coefficient(features[col], target)
        if np.isfinite(ic) and abs(ic) >= min_abs_ic:
            selected.append(col)
    return selected
