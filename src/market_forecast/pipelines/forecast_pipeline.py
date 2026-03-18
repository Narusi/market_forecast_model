from __future__ import annotations

import pandas as pd

from market_forecast.config.schemas import ForecastConfig
from market_forecast.data.preprocess import returns_from_prices
from market_forecast.data.validators import validate_prices
from market_forecast.ensemble.combiner import equal_weight, performance_weighted
from market_forecast.features.seasonality import seasonality_features
from market_forecast.features.selection import select_predictive_features
from market_forecast.features.technical import technical_features
from market_forecast.features.trend import trend_features
from market_forecast.features.volatility import volatility_features
from market_forecast.models.arima import ArimaForecaster
from market_forecast.models.garch import EgarchVolatilityModel
from market_forecast.models.svr import SvrForecaster
from market_forecast.risk.analytics import regime_distribution, risk_summary
from market_forecast.signals.technical_tiers import tiered_signal_decision
from market_forecast.signals.transform import forecast_to_signal


class ForecastPipeline:
    def __init__(self, config: ForecastConfig) -> None:
        self.config = config
        self.prices: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.target_col: str | None = None
        self.selected_features: list[str] = []
        self.models: dict[str, object] = {}
        self.model_scores: dict[str, float] = {"arima": 1.0, "svr": 1.0, "egarch": 0.5}
        self.feature_frame: pd.DataFrame | None = None

    def _build_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        feat = trend_features(prices)
        if self.config.feature.include_seasonality:
            feat = feat.join(seasonality_features(prices.index), how="left")
        if self.config.feature.include_volatility:
            feat = feat.join(volatility_features(returns), how="left")
        if self.config.feature.include_technical and self.config.mode == "financial":
            feat = feat.join(technical_features(prices), how="left")
        return feat

    def fit(self, prices: pd.DataFrame, target_col: str | None = None) -> "ForecastPipeline":
        prices = validate_prices(prices)
        self.prices = prices
        self.returns = returns_from_prices(prices)
        self.target_col = target_col or prices.columns[0]

        features = self._build_features(prices, self.returns)
        target = self.returns[self.target_col]
        selected = select_predictive_features(features, target, self.config.feature.min_feature_ic)
        self.selected_features = selected or list(features.columns[: min(20, len(features.columns))])
        self.feature_frame = features[self.selected_features]

        if self.config.model.use_arima:
            self.models["arima"] = ArimaForecaster(order=self.config.model.arima_order).fit(target)

        if self.config.model.use_svr:
            self.models["svr"] = SvrForecaster(
                c=self.config.model.svr_c, epsilon=self.config.model.svr_epsilon
            ).fit(target, self.feature_frame)

        if self.config.model.use_egarch:
            self.models["egarch"] = EgarchVolatilityModel(
                p=self.config.model.egarch_p, q=self.config.model.egarch_q
            ).fit(target)

        return self

    def predict(self, horizons: list[int]) -> pd.DataFrame:
        if self.returns is None or self.feature_frame is None or self.target_col is None:
            raise RuntimeError("Pipeline is not fitted")

        rows = []
        for h in horizons:
            model_preds: dict[str, float] = {}
            if "arima" in self.models:
                model_preds["arima"] = self.models["arima"].predict(horizon=h)
            if "svr" in self.models:
                model_preds["svr"] = self.models["svr"].predict(horizon=h, X_future=self.feature_frame)
            if "egarch" in self.models:
                vol = self.models["egarch"].forecast_volatility(horizon=max(h, 1))
                model_preds["egarch"] = -vol

            if self.config.ensemble.method == "performance_weighted":
                ensemble_pred = performance_weighted(model_preds, self.model_scores)
            else:
                ensemble_pred = equal_weight(model_preds)

            rows.append({"horizon_weeks": h, "prediction": ensemble_pred, **model_preds})

        return pd.DataFrame(rows)

    def generate_signals(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if self.prices is None or self.target_col is None:
            raise RuntimeError("Pipeline is not fitted")

        out = forecast_df.copy()
        out["signal"] = out["prediction"].apply(
            lambda x: forecast_to_signal(
                x,
                buy_threshold=self.config.signal.buy_threshold,
                sell_threshold=self.config.signal.sell_threshold,
            )
        )

        arima_default = float(out.get("arima", pd.Series([0.0])).iloc[0])
        tier = tiered_signal_decision(self.prices, self.target_col, arima_pred=arima_default)
        out["regime"] = tier["current_regime"]
        out["tier1_signal"] = "buy" if tier["tier1_vote"] > 0 else "sell" if tier["tier1_vote"] < 0 else "hold"
        out["tier2_signal"] = "buy" if tier["tier2_vote"] > 0 else "sell" if tier["tier2_vote"] < 0 else "hold"
        out["tier3_signal"] = tier["tier3_signal"]
        out["tier1_indicators"] = ",".join(tier["tier1_indicators"])

        return out

    def summarize_current_state(self) -> dict[str, object]:
        if self.prices is None or self.returns is None or self.target_col is None:
            raise RuntimeError("Pipeline is not fitted")

        target_returns = self.returns[self.target_col].dropna()
        risk = risk_summary(self.returns)
        regime = regime_distribution(target_returns)
        arima_pred = 0.0
        if "arima" in self.models:
            arima_pred = float(self.models["arima"].predict(horizon=1))
        tiers = tiered_signal_decision(self.prices, self.target_col, arima_pred=arima_pred)

        return {
            "target_asset": self.target_col,
            "risk": risk,
            "regime": regime,
            "signals": {
                "tier1": "buy" if tiers["tier1_vote"] > 0 else "sell" if tiers["tier1_vote"] < 0 else "hold",
                "tier2": "buy" if tiers["tier2_vote"] > 0 else "sell" if tiers["tier2_vote"] < 0 else "hold",
                "tier3": tiers["tier3_signal"],
                "selected_tier1_indicators": tiers["tier1_indicators"],
            },
        }
