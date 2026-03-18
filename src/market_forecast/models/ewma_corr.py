from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    x = returns.dropna().astype(float)
    if x.empty:
        raise ValueError("No returns available for EWMA covariance")
    n = x.shape[1]
    cov = np.cov(x.iloc[: min(20, len(x))].T)
    cov = np.atleast_2d(cov)
    if cov.shape != (n, n):
        cov = np.eye(n) * float(np.var(x.values))
    for i in range(min(20, len(x)), len(x)):
        r = x.iloc[i].values.reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
    return pd.DataFrame(cov, index=x.columns, columns=x.columns)


def ewma_correlation(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    cov = ewma_covariance(returns, lam=lam)
    std = np.sqrt(np.diag(cov.values))
    denom = np.outer(std, std)
    corr = cov.values / np.where(denom == 0, np.nan, denom)
    corr = np.nan_to_num(corr)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
