from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from market_forecast.models.base import BaseForecaster


class ArimaForecaster(BaseForecaster):
    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = order
        self._fit_res = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "ArimaForecaster":
        model = ARIMA(endog=y.astype(float), exog=X, order=self.order)
        self._fit_res = model.fit()
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> float:
        if self._fit_res is None:
            raise RuntimeError("Model is not fitted")
        forecast = self._fit_res.forecast(steps=horizon, exog=X_future)
        return float(forecast.iloc[-1])
