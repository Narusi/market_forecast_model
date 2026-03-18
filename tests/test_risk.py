import numpy as np
import pandas as pd

from market_forecast.models.ewma_corr import ewma_correlation


def test_ewma_correlation_shape():
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(120, 3)), index=idx, columns=["A", "B", "C"])
    corr = ewma_correlation(df)
    assert corr.shape == (3, 3)
