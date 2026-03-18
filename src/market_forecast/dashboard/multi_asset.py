from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    return px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="EWMA Correlation Heatmap",
        aspect="auto",
    )


def asset_signal_matrix(signals: pd.DataFrame) -> go.Figure:
    if signals.empty:
        return px.imshow(pd.DataFrame([[0.0]]), text_auto=True, title="Asset Signal Matrix")

    matrix = signals.copy()
    if "asset" in matrix.columns:
        matrix = matrix.set_index("asset")
    if "timestamp" in matrix.columns:
        matrix = matrix.drop(columns=["timestamp"])

    numeric = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return px.imshow(
        numeric,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        title="Asset Signal Matrix",
        aspect="auto",
    )


def portfolio_risk_decomposition(
    returns: pd.DataFrame,
    state_summary: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(columns=["asset", "weight", "volatility", "risk_contribution"])  # pragma: no cover

    vol = returns.std(ddof=0)
    inv_vol = 1.0 / vol.replace(0, pd.NA)
    inv_vol = inv_vol.fillna(0.0)
    if float(inv_vol.sum()) == 0:
        weights = pd.Series(1.0 / len(inv_vol), index=inv_vol.index)
    else:
        weights = inv_vol / float(inv_vol.sum())

    weighted_vol = weights * vol
    total = float(weighted_vol.sum()) if float(weighted_vol.sum()) != 0 else 1.0
    contrib = weighted_vol / total

    out = pd.DataFrame(
        {
            "asset": vol.index,
            "weight": weights.values,
            "volatility": vol.values,
            "risk_contribution": contrib.values,
        }
    ).sort_values("risk_contribution", ascending=False)

    if state_summary and isinstance(state_summary, Mapping):
        risk_meta = state_summary.get("risk", {})
        out.attrs["portfolio_risk_level"] = risk_meta.get("risk_level", "unknown")

    return out.reset_index(drop=True)
