from __future__ import annotations

import numpy as np
import pandas as pd

from market_forecast.ensemble.regime_weighting import current_regime, regime_series


INDICATOR_CATEGORIES = {
    "rsi_14": "mean_reversion",
    "stoch_kd": "mean_reversion",
    "bb_pctb": "mean_reversion",
    "cci_20": "mean_reversion",
    "williams_r": "mean_reversion",
    "macd_cross": "trend",
    "ema_spread_20_50": "trend",
    "mom_10": "momentum",
    "roc_12": "momentum",
}


def _sign(x: pd.Series, threshold: float = 0.0) -> pd.Series:
    return np.where(x > threshold, 1, np.where(x < -threshold, -1, 0))


def indicator_signal_frame(prices: pd.DataFrame, asset: str) -> pd.DataFrame:
    s = prices[asset].astype(float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))

    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()

    low14 = s.rolling(14).min()
    high14 = s.rolling(14).max()
    stoch_k = 100 * (s - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    sma20 = s.rolling(20).mean()
    sd20 = s.rolling(20).std()
    bb_upper = sma20 + 2 * sd20
    bb_lower = sma20 - 2 * sd20
    bb_pctb = (s - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    ema20 = s.ewm(span=20, adjust=False).mean()
    ema50 = s.ewm(span=50, adjust=False).mean()
    ema_spread = ema20 / ema50 - 1.0
    mom10 = s / s.shift(10) - 1.0
    roc12 = s.pct_change(12)

    mad = (s - s.rolling(20).mean()).abs().rolling(20).mean()
    cci20 = (s - s.rolling(20).mean()) / (0.015 * mad).replace(0, np.nan)
    wr = -100 * (high14 - s) / (high14 - low14).replace(0, np.nan)

    signals = pd.DataFrame(index=prices.index)
    signals["rsi_14"] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    signals["macd_cross"] = _sign(macd - macd_sig, threshold=0.0)
    signals["stoch_kd"] = np.where((stoch_k < 20) & (stoch_k > stoch_d), 1, np.where((stoch_k > 80) & (stoch_k < stoch_d), -1, 0))
    signals["bb_pctb"] = np.where(bb_pctb < 0, 1, np.where(bb_pctb > 1, -1, 0))
    signals["ema_spread_20_50"] = _sign(ema_spread, threshold=0.0)
    signals["mom_10"] = _sign(mom10, threshold=0.0)
    signals["roc_12"] = _sign(roc12, threshold=0.0)
    signals["cci_20"] = np.where(cci20 < -100, 1, np.where(cci20 > 100, -1, 0))
    signals["williams_r"] = np.where(wr < -80, 1, np.where(wr > -20, -1, 0))
    return signals.fillna(0).astype(int)


def _vote(df: pd.DataFrame) -> pd.Series:
    raw = df.sum(axis=1)
    return pd.Series(np.where(raw > 0, 1, np.where(raw < 0, -1, 0)), index=df.index)


def evaluate_indicator_scores_by_regime(prices: pd.DataFrame, asset: str) -> pd.DataFrame:
    ret_fwd = prices[asset].pct_change().shift(-1)
    signals = indicator_signal_frame(prices, asset)
    regimes = regime_series(prices[asset].pct_change())

    rows = []
    for ind in signals.columns:
        pnl = signals[ind] * ret_fwd
        for reg in regimes.dropna().unique():
            mask = regimes == reg
            sample = pnl[mask].dropna()
            score = float(sample.mean()) if not sample.empty else 0.0
            hit = float((sample > 0).mean()) if not sample.empty else 0.0
            rows.append({"indicator": ind, "regime": reg, "score": score, "hit_rate": hit})
    return pd.DataFrame(rows)


def _select_tier1(score_df: pd.DataFrame, reg: str) -> list[str]:
    selected: list[str] = []
    cur = score_df[score_df["regime"] == reg]
    for cat in sorted(set(INDICATOR_CATEGORIES.values())):
        inds = [k for k, v in INDICATOR_CATEGORIES.items() if v == cat]
        sub = cur[cur["indicator"].isin(inds)].sort_values(["score", "hit_rate"], ascending=False)
        if not sub.empty:
            selected.append(str(sub.iloc[0]["indicator"]))
    return selected


def _select_tier2(score_df: pd.DataFrame, reg: str) -> dict[str, list[str]]:
    cur = score_df[score_df["regime"] == reg]
    out: dict[str, list[str]] = {}
    for cat in sorted(set(INDICATOR_CATEGORIES.values())):
        inds = [k for k, v in INDICATOR_CATEGORIES.items() if v == cat]
        sub = cur[cur["indicator"].isin(inds)].sort_values(["score", "hit_rate"], ascending=False)
        out[cat] = list(sub["indicator"].head(3))
    return out


def tiered_signal_decision(prices: pd.DataFrame, asset: str, arima_pred: float, threshold: float = 0.0) -> dict[str, object]:
    signals = indicator_signal_frame(prices, asset)
    scores = evaluate_indicator_scores_by_regime(prices, asset)
    reg = current_regime(prices[asset].pct_change())

    tier1_inds = _select_tier1(scores, reg)
    t1_vote = int(_vote(signals[tier1_inds]).iloc[-1]) if tier1_inds else 0

    tier2_map = _select_tier2(scores, reg)
    tier2_category_votes = {}
    cat_votes = []
    for cat, inds in tier2_map.items():
        if inds:
            v = int(_vote(signals[inds]).iloc[-1])
            tier2_category_votes[cat] = v
            cat_votes.append(v)
    tier2_vote = int(np.sign(sum(cat_votes))) if cat_votes else 0

    arima_vote = int(np.sign(arima_pred - threshold))
    if tier2_vote == 0:
        tier3_label = "hold"
    elif tier2_vote == arima_vote and tier2_vote != 0:
        tier3_label = "strong_buy" if tier2_vote > 0 else "strong_sell"
    elif arima_vote == 0:
        tier3_label = "buy" if tier2_vote > 0 else "sell"
    else:
        tier3_label = "weak_buy" if tier2_vote > 0 else "weak_sell"

    return {
        "current_regime": reg,
        "tier1_indicators": tier1_inds,
        "tier1_vote": t1_vote,
        "tier2_category_votes": tier2_category_votes,
        "tier2_vote": tier2_vote,
        "arima_vote": arima_vote,
        "tier3_signal": tier3_label,
    }
