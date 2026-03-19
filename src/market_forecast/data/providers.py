from __future__ import annotations

import pandas as pd


def fetch_macro_data(
    start: str = "1990-01-01",
    end: str | None = None,
    series: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch macroeconomic time series from FRED when available.

    Parameters
    ----------
    start:
        Start date in YYYY-MM-DD format.
    end:
        End date in YYYY-MM-DD format. If None, current date is used by the provider.
    series:
        Mapping of output column names to FRED series IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date. Returns empty DataFrame if provider dependencies
        are not installed or retrieval fails.
    """
    ids = series or {
        "fed_funds": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "unemployment": "UNRATE",
    }

    try:
        from pandas_datareader import data as web  # type: ignore

        frames = []
        for name, sid in ids.items():
            s = web.DataReader(sid, "fred", start=start, end=end)
            s.columns = [name]
            frames.append(s)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index().ffill()
    except Exception:
        return pd.DataFrame()
