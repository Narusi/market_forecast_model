from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from market_forecast.models.base import BaseForecaster


class SvrForecaster(BaseForecaster):
    def __init__(self, c: float = 1.0, epsilon: float = 0.1) -> None:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("svr", SVR(C=c, epsilon=epsilon, kernel="rbf")),
            ]
        )

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "SvrForecaster":
        if X is None:
            raise ValueError("SVR requires feature matrix X")
        aligned = pd.concat([X, y.rename("target")], axis=1).dropna()
        self.pipeline.fit(aligned[X.columns], aligned["target"])
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> float:
        if X_future is None or X_future.empty:
            raise ValueError("X_future is required for SVR prediction")
        pred = self.pipeline.predict(X_future.tail(1))[0]
        return float(pred)
