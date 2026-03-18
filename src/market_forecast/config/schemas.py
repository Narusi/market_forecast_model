from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    include_technical: bool = True
    include_seasonality: bool = True
    include_volatility: bool = True
    min_feature_ic: float = 0.01


class ModelConfig(BaseModel):
    use_arima: bool = True
    use_egarch: bool = True
    use_svr: bool = True
    arima_order: tuple[int, int, int] = (1, 1, 1)
    egarch_p: int = 1
    egarch_q: int = 1
    svr_c: float = 1.0
    svr_epsilon: float = 0.1


class EnsembleConfig(BaseModel):
    method: Literal["equal", "performance_weighted"] = "performance_weighted"


class SignalConfig(BaseModel):
    buy_threshold: float = 0.01
    sell_threshold: float = -0.01


class BacktestConfig(BaseModel):
    train_window: int = 252
    test_window: int = 20
    step: int = 20


class ForecastConfig(BaseModel):
    mode: Literal["financial", "generic_ts"] = "financial"
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
