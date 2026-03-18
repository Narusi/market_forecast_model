from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecast.dashboard.multi_asset import (
    asset_signal_matrix,
    correlation_heatmap,
    portfolio_risk_decomposition,
)
from market_forecast.dashboard.single_asset import (
    forecast_horizon_table,
    risk_regime_distribution_card,
    single_asset_figure,
    tiered_signal_card,
)
from market_forecast.risk.analytics import correlation_matrix


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        loaded = json.loads(path.read_text())
        return pd.DataFrame(loaded)
    raise ValueError(f"Unsupported artifact format: {path}")


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_single_asset_dashboard(
    price: pd.Series,
    forecast: pd.Series,
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
    state_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timeseries": single_asset_figure(price, forecast),
        "forecast_table": forecast_horizon_table(predictions),
        "signal_card": tiered_signal_card(signals),
        "risk_card": risk_regime_distribution_card(state_summary),
    }


def build_multi_asset_dashboard(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    state_summary: dict[str, Any],
) -> dict[str, Any]:
    corr = correlation_matrix(returns)
    return {
        "corr_heatmap": correlation_heatmap(corr),
        "signal_matrix": asset_signal_matrix(signals),
        "risk_decomp": portfolio_risk_decomposition(returns, state_summary),
    }


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Market Forecast Dashboard", layout="wide")
    st.title("Market Forecast Dashboard")

    mode = st.sidebar.radio("Mode", options=["Single Asset", "Multi Asset"], index=0)

    st.sidebar.subheader("Serialized artifacts")
    predictions_path = Path(st.sidebar.text_input("Predictions path", "artifacts/predictions.csv"))
    signals_path = Path(st.sidebar.text_input("Signals path", "artifacts/signals.csv"))
    state_path = Path(st.sidebar.text_input("State summary path", "artifacts/state_summary.json"))

    predictions = _load_table(predictions_path)
    signals = _load_table(signals_path)
    state_summary = _load_summary(state_path)

    if mode == "Single Asset":
        price_col = st.sidebar.text_input("Price column", "price")
        forecast_col = st.sidebar.text_input("Forecast column", "forecast")
        series_df = predictions.copy()

        if "date" in series_df.columns:
            series_df["date"] = pd.to_datetime(series_df["date"])
            series_df = series_df.set_index("date")

        if price_col not in series_df.columns or forecast_col not in series_df.columns:
            st.error(
                f"Predictions artifact must include '{price_col}' and '{forecast_col}' columns for single-asset mode."
            )
            return

        views = build_single_asset_dashboard(
            price=series_df[price_col],
            forecast=series_df[forecast_col],
            predictions=predictions,
            signals=signals,
            state_summary=state_summary,
        )

        st.plotly_chart(views["timeseries"], use_container_width=True)
        st.subheader("Forecast Horizon Table")
        st.dataframe(views["forecast_table"], use_container_width=True)

        signal = views["signal_card"]
        st.subheader("Tiered Signal Card")
        st.markdown(
            f"**Overall Signal:** :{_streamlit_color(signal['overall_color'])}[{signal['overall'].upper()}]"
        )
        st.write(
            {
                "tier1": signal["tier1"],
                "tier2": signal["tier2"],
                "tier3": signal["tier3"],
                "regime": signal["regime"],
                "tier1_indicators": signal["tier1_indicators"],
            }
        )

        st.subheader("Risk/Regime Distribution")
        st.dataframe(views["risk_card"], use_container_width=True)
    else:
        returns_path = Path(st.sidebar.text_input("Returns path", "artifacts/returns.csv"))
        returns = _load_table(returns_path)
        if "date" in returns.columns:
            returns["date"] = pd.to_datetime(returns["date"])
            returns = returns.set_index("date")

        views = build_multi_asset_dashboard(returns=returns, signals=signals, state_summary=state_summary)
        st.subheader("EWMA Correlation Heatmap")
        st.plotly_chart(views["corr_heatmap"], use_container_width=True)

        st.subheader("Asset Signal Matrix")
        st.plotly_chart(views["signal_matrix"], use_container_width=True)

        st.subheader("Portfolio Risk Decomposition")
        st.dataframe(views["risk_decomp"], use_container_width=True)


def _streamlit_color(hex_color: str) -> str:
    if hex_color == "#16a34a":
        return "green"
    if hex_color == "#dc2626":
        return "red"
    return "gray"


if __name__ == "__main__":
    run_streamlit_app()
