from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

_SIGNAL_COLORS = {
    "buy": "#16a34a",
    "hold": "#64748b",
    "sell": "#dc2626",
}


def single_asset_figure(price: pd.Series, forecast: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            name="Price",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name="Forecast",
            line={"color": "#f59e0b", "width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Single Asset Price vs Forecast",
        template="plotly_white",
        margin={"l": 30, "r": 20, "t": 60, "b": 30},
        legend={"orientation": "h", "y": 1.05},
        yaxis_title="Level",
    )
    return fig


def forecast_horizon_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required = ["horizon_weeks", "prediction"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing required columns for forecast table: {missing}")

    view = predictions.copy()
    view = view.sort_values("horizon_weeks")
    view["prediction"] = view["prediction"].astype(float)
    view["predicted_move_bps"] = (view["prediction"] * 10_000).round(2)
    return view[["horizon_weeks", "prediction", "predicted_move_bps"]]


def tiered_signal_card(signals: pd.DataFrame) -> dict[str, str]:
    if signals.empty:
        return {
            "overall": "hold",
            "tier1": "hold",
            "tier2": "hold",
            "tier3": "hold",
            "regime": "unknown",
        }

    latest = signals.iloc[-1]
    overall = str(latest.get("signal", "hold"))
    return {
        "overall": overall,
        "overall_color": _SIGNAL_COLORS.get(overall, "#64748b"),
        "tier1": str(latest.get("tier1_signal", "hold")),
        "tier2": str(latest.get("tier2_signal", "hold")),
        "tier3": str(latest.get("tier3_signal", "hold")),
        "regime": str(latest.get("regime", "unknown")),
        "tier1_indicators": str(latest.get("tier1_indicators", "")),
    }


def risk_regime_distribution_card(state_summary: Mapping[str, object]) -> pd.DataFrame:
    risk = state_summary.get("risk", {}) if isinstance(state_summary, Mapping) else {}
    regime = state_summary.get("regime", {}) if isinstance(state_summary, Mapping) else {}

    rows = [
        ("risk_level", risk.get("risk_level", "unknown")),
        ("avg_volatility", risk.get("avg_volatility", 0.0)),
        ("recent_volatility", risk.get("recent_volatility", 0.0)),
        ("max_drawdown_proxy", risk.get("max_drawdown_proxy", 0.0)),
        ("regime", regime.get("regime", "unknown")),
        ("lookback_days", regime.get("lookback_days", 0)),
        ("mean_return", regime.get("mean_return", 0.0)),
        ("volatility", regime.get("volatility", 0.0)),
        ("skewness", regime.get("skewness", 0.0)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
