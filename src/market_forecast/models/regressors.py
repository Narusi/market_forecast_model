from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


class RidgeForecaster:
    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)
        self.columns: list[str] = []

    def fit(self, y: pd.Series, X: pd.DataFrame) -> "RidgeForecaster":
        aligned = pd.concat([X, y.rename("target")], axis=1).dropna()
        self.columns = list(X.columns)
        self.model.fit(aligned[self.columns], aligned["target"])
        return self

    def predict(self, X_future: pd.DataFrame) -> float:
        return float(self.model.predict(X_future[self.columns].tail(1))[0])
