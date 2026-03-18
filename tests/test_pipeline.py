import numpy as np
import pandas as pd

from market_forecast.config.schemas import ForecastConfig
from market_forecast.pipelines.forecast_pipeline import ForecastPipeline


def make_prices(n: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    r1 = rng.normal(0.0005, 0.01, size=n)
    r2 = rng.normal(0.0003, 0.008, size=n)
    p1 = 100 * np.cumprod(1 + r1)
    p2 = 80 * np.cumprod(1 + r2)
    return pd.DataFrame({"SPY": p1, "QQQ": p2}, index=idx)


def test_pipeline_fit_predict_signals():
    prices = make_prices()
    cfg = ForecastConfig()
    pipe = ForecastPipeline(cfg).fit(prices, target_col="SPY")
    pred = pipe.predict([1, 2, 4, 8, 12])
    out = pipe.generate_signals(pred)

    assert len(pred) == 5
    assert {"horizon_weeks", "prediction"}.issubset(set(pred.columns))
    assert set(out["signal"].unique()).issubset({"buy", "hold", "sell"})
    assert {"tier1_signal", "tier2_signal", "tier3_signal", "regime"}.issubset(set(out.columns))
    state = pipe.summarize_current_state()
    assert "risk" in state and "regime" in state and "signals" in state
