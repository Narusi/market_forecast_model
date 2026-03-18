from __future__ import annotations

import numpy as np
import pandas as pd


def detect_regime(trend_ratio: float, volatility: float) -> str:
    if volatility > 0.03 and trend_ratio >= 0:
        return "high_vol_up"
    if volatility > 0.03 and trend_ratio < 0:
        return "high_vol_down"
    if trend_ratio > 0.005:
        return "uptrend"
    if trend_ratio < -0.005:
        return "downtrend"
    return "sideways"


def regime_series(returns: pd.Series, trend_window: int = 20, vol_window: int = 20) -> pd.Series:
    r = returns.fillna(0.0).astype(float)
    trend = r.rolling(trend_window).mean()
    vol = r.rolling(vol_window).std()
    labels = [detect_regime(float(t), float(v)) for t, v in zip(trend, vol)]
    return pd.Series(labels, index=returns.index)


def current_regime(returns: pd.Series) -> str:
    series = regime_series(returns)
    return str(series.iloc[-1]) if not series.empty else "sideways"


def regime_weight(score_by_regime: dict[str, float], current: str) -> float:
    base = float(score_by_regime.get(current, 0.0))
    non_zero = [abs(v) for v in score_by_regime.values() if v != 0]
    denom = float(np.mean(non_zero)) if non_zero else 1.0
    if denom == 0:
        return 0.0
    return base / denom
