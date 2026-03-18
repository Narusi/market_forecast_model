from __future__ import annotations

import pandas as pd


def contribution_report(pred_by_model: pd.DataFrame) -> pd.Series:
    return pred_by_model.std().sort_values(ascending=False)
