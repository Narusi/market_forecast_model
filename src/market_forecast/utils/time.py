from __future__ import annotations

import pandas as pd


def enforce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame index must be a pandas.DatetimeIndex")
    return df.sort_index()
