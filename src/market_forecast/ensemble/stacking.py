from __future__ import annotations

import numpy as np


def simple_stack(pred_matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return pred_matrix.mean(axis=1)
    norm = weights / weights.sum()
    return pred_matrix @ norm
