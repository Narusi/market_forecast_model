from __future__ import annotations

import pandas as pd


def merge_macro_features(features: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    if macro is None or macro.empty:
        return features
    return features.join(macro, how="left").ffill()
