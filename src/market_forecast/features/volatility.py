from __future__ import annotations

import pandas as pd


def volatility_features(returns: pd.DataFrame, windows: tuple[int, int] = (10, 20)) -> pd.DataFrame:
    feats = {}
    for col in returns.columns:
        for w in windows:
            feats[f"{col}_rv_{w}"] = returns[col].rolling(w).std()
    return pd.DataFrame(feats, index=returns.index)
