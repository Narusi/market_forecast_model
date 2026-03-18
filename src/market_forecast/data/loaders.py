from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, date_col: str = "Date", index_col: str | None = None) -> pd.DataFrame:
    idx = index_col or date_col
    df = pd.read_csv(path, parse_dates=[idx])
    df = df.set_index(idx)
    return df.sort_index()


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Parquet file must contain a DatetimeIndex")
    return df.sort_index()
