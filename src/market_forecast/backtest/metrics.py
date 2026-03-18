from __future__ import annotations

import numpy as np
import pandas as pd


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    if aligned.empty:
        return 0.0
    return float((np.sign(aligned["t"]) == np.sign(aligned["p"]).astype(int)).mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    if aligned.empty:
        return 0.0
    return float(np.sqrt(((aligned["t"] - aligned["p"]) ** 2).mean()))
