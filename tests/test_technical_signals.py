import numpy as np
import pandas as pd

from market_forecast.features.technical import technical_features
from market_forecast.signals.technical_tiers import indicator_signal_frame, tiered_signal_decision


def make_prices(n: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    rng = np.random.default_rng(123)
    r = rng.normal(0.0004, 0.012, size=n)
    px = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"SPY": px}, index=idx)


def test_technical_features_include_expected_indicators():
    prices = make_prices()
    feats = technical_features(prices)
    expected = {
        "SPY_rsi_14",
        "SPY_macd",
        "SPY_macd_signal",
        "SPY_stoch_k",
        "SPY_bb_pctb",
        "SPY_ema_spread_20_50",
        "SPY_mom_10",
        "SPY_roc_12",
        "SPY_cci_20",
        "SPY_williams_r",
    }
    assert expected.issubset(set(feats.columns))


def test_tiered_signal_decision_outputs_expected_keys():
    prices = make_prices()
    sig = indicator_signal_frame(prices, "SPY")
    assert sig.shape[1] >= 9

    decision = tiered_signal_decision(prices, "SPY", arima_pred=0.01)
    assert decision["tier3_signal"] in {"hold", "buy", "sell", "strong_buy", "strong_sell", "weak_buy", "weak_sell"}
    assert isinstance(decision["tier1_indicators"], list)
