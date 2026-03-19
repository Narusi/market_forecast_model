from __future__ import annotations

import pandas as pd
from arch import arch_model


class EgarchVolatilityModel:
    def __init__(self, p: int = 1, q: int = 1) -> None:
        self.p = p
        self.q = q
        self._fit_res = None

    def fit(self, returns: pd.Series) -> "EgarchVolatilityModel":
        y = (returns.dropna().astype(float) * 100.0).clip(-20, 20)
        model = arch_model(y, vol="EGARCH", p=self.p, q=self.q, mean="Zero", dist="normal")
        self._fit_res = model.fit(disp="off")
        return self

    def forecast_volatility(self, horizon: int = 1) -> float:
        if self._fit_res is None:
            raise RuntimeError("Model is not fitted")
        f = self._fit_res.forecast(horizon=horizon)
        var = float(f.variance.iloc[-1, -1])
        return (var**0.5) / 100.0
