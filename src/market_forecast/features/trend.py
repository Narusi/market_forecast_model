from __future__ import annotations

import pandas as pd


def trend_features(prices: pd.DataFrame, short: int = 10, long: int = 30) -> pd.DataFrame:
    feats = {}
    for col in prices.columns:
        s = prices[col]
        feats[f"{col}_sma_{short}"] = s.rolling(short).mean()
        feats[f"{col}_sma_{long}"] = s.rolling(long).mean()
        feats[f"{col}_trend_ratio"] = feats[f"{col}_sma_{short}"] / feats[f"{col}_sma_{long}"] - 1.0
    return pd.DataFrame(feats, index=prices.index)
