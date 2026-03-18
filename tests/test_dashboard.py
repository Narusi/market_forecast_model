import numpy as np
import pandas as pd

from market_forecast.dashboard.app import build_multi_asset_dashboard, build_single_asset_dashboard


def _sample_data(n: int = 64):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    price = pd.Series(100 + np.cumsum(rng.normal(0, 1, size=n)), index=idx)
    forecast = pd.Series(100 + np.cumsum(rng.normal(0, 1, size=n)), index=idx)
    predictions = pd.DataFrame(
        {
            "horizon_weeks": [1, 2, 4, 8, 12],
            "prediction": rng.normal(0, 0.01, size=5),
            "price": [float(price.iloc[-1])] * 5,
            "forecast": [float(forecast.iloc[-1])] * 5,
        }
    )
    signals = pd.DataFrame(
        {
            "signal": ["buy"],
            "tier1_signal": ["buy"],
            "tier2_signal": ["hold"],
            "tier3_signal": ["buy"],
            "regime": ["normal"],
            "tier1_indicators": ["rsi_14,macd_cross"],
        }
    )
    returns = pd.DataFrame(
        {
            "SPY": rng.normal(0, 0.01, size=n),
            "QQQ": rng.normal(0, 0.013, size=n),
            "TLT": rng.normal(0, 0.006, size=n),
        },
        index=idx,
    )
    state = {
        "risk": {"risk_level": "normal", "avg_volatility": 0.01, "recent_volatility": 0.012},
        "regime": {"regime": "risk_on", "lookback_days": 63, "mean_return": 0.001, "volatility": 0.01},
    }
    return price, forecast, predictions, signals, returns, state


def test_single_asset_dashboard_smoke():
    price, forecast, predictions, signals, _, state = _sample_data()
    views = build_single_asset_dashboard(price, forecast, predictions, signals, state)
    assert {"timeseries", "forecast_table", "signal_card", "risk_card"}.issubset(set(views.keys()))
    assert not views["forecast_table"].empty


def test_multi_asset_dashboard_smoke():
    _, _, _, signals, returns, state = _sample_data()
    matrix = pd.DataFrame(
        {
            "asset": ["SPY", "QQQ", "TLT"],
            "h1_signal": [1, -1, 0],
            "h4_signal": [1, 0, -1],
        }
    )
    views = build_multi_asset_dashboard(returns, matrix, state)
    assert {"corr_heatmap", "signal_matrix", "risk_decomp"}.issubset(set(views.keys()))
    assert not views["risk_decomp"].empty
