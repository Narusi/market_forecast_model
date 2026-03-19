from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Close-price technical indicators for each asset column."""
    feats: dict[str, pd.Series] = {}
    for col in prices.columns:
        s = prices[col].astype(float)
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        # RSI
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        feats[f"{col}_rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = _ema(s, 12)
        ema26 = _ema(s, 26)
        macd = ema12 - ema26
        macd_sig = _ema(macd, 9)
        feats[f"{col}_macd"] = macd
        feats[f"{col}_macd_signal"] = macd_sig
        feats[f"{col}_macd_hist"] = macd - macd_sig

        # Stochastic proxy from close-only range
        roll_low = s.rolling(14).min()
        roll_high = s.rolling(14).max()
        stoch_k = 100 * (s - roll_low) / (roll_high - roll_low).replace(0, np.nan)
        stoch_d = stoch_k.rolling(3).mean()
        feats[f"{col}_stoch_k"] = stoch_k
        feats[f"{col}_stoch_d"] = stoch_d

        # Bollinger
        sma20 = s.rolling(20).mean()
        sd20 = s.rolling(20).std()
        feats[f"{col}_bb_upper"] = sma20 + 2 * sd20
        feats[f"{col}_bb_lower"] = sma20 - 2 * sd20
        feats[f"{col}_bb_pctb"] = (s - feats[f"{col}_bb_lower"]) / (
            feats[f"{col}_bb_upper"] - feats[f"{col}_bb_lower"]
        ).replace(0, np.nan)

        # Trend and momentum
        ema20 = _ema(s, 20)
        ema50 = _ema(s, 50)
        feats[f"{col}_ema_spread_20_50"] = ema20 / ema50 - 1.0
        feats[f"{col}_mom_10"] = s / s.shift(10) - 1.0
        feats[f"{col}_roc_12"] = s.pct_change(12)

        # Mean-reversion oscillators
        tp = s
        mad = (tp - tp.rolling(20).mean()).abs().rolling(20).mean()
        feats[f"{col}_cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad).replace(0, np.nan)
        hh = s.rolling(14).max()
        ll = s.rolling(14).min()
        feats[f"{col}_williams_r"] = -100 * (hh - s) / (hh - ll).replace(0, np.nan)

    return pd.DataFrame(feats, index=prices.index)
