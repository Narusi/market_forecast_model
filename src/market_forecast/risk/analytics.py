from __future__ import annotations

import pandas as pd

from market_forecast.ensemble.regime_weighting import current_regime
from market_forecast.models.ewma_corr import ewma_correlation


def risk_summary(returns: pd.DataFrame) -> dict[str, float | str]:
    vol_series = returns.std(axis=1) if isinstance(returns, pd.DataFrame) else returns
    avg_vol = float(returns.std().mean())
    dd = (returns.cumsum() - returns.cumsum().cummax()).min().min()
    recent_vol = float(vol_series.tail(20).mean()) if len(vol_series) >= 20 else float(vol_series.mean())
    hist_vol = float(vol_series.mean()) if len(vol_series) else 0.0

    if recent_vol > hist_vol * 1.5:
        risk_level = "high"
    elif recent_vol > hist_vol * 1.1:
        risk_level = "elevated"
    else:
        risk_level = "normal"

    return {
        "avg_volatility": avg_vol,
        "recent_volatility": recent_vol,
        "max_drawdown_proxy": float(dd),
        "risk_level": risk_level,
    }


def regime_distribution(returns: pd.Series) -> dict[str, float | str]:
    reg = current_regime(returns)
    tail = returns.tail(63).dropna()
    mu = float(tail.mean()) if not tail.empty else 0.0
    sd = float(tail.std(ddof=0)) if not tail.empty else 0.0
    sk = float(tail.skew()) if len(tail) > 2 else 0.0
    return {
        "regime": reg,
        "lookback_days": int(len(tail)),
        "mean_return": mu,
        "volatility": sd,
        "skewness": sk,
    }


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return ewma_correlation(returns)
