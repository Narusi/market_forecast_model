from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_predictions(pred: pd.Series) -> pd.Series:
    sd = pred.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pred * 0
    return (pred - pred.mean()) / sd
