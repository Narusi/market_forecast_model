from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "BaseForecaster":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> float:
        raise NotImplementedError
