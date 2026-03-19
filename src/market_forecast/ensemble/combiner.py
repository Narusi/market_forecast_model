from __future__ import annotations

import numpy as np


def equal_weight(predictions: dict[str, float]) -> float:
    return float(np.mean(list(predictions.values())))


def performance_weighted(predictions: dict[str, float], scores: dict[str, float]) -> float:
    names = [k for k in predictions if k in scores]
    if not names:
        return equal_weight(predictions)
    raw = np.array([max(scores[n], 1e-6) for n in names], dtype=float)
    w = raw / raw.sum()
    p = np.array([predictions[n] for n in names], dtype=float)
    return float((w * p).sum())
