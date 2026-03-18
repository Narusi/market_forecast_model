from __future__ import annotations

import pandas as pd


def fetch_macro_stub() -> pd.DataFrame:
    """Fallback macro provider. Replace with FRED/other providers in production."""
    return pd.DataFrame()
