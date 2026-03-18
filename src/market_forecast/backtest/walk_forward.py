from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train_idx: pd.Index
    test_idx: pd.Index


def walk_forward_splits(index: pd.Index, train_window: int, test_window: int, step: int) -> list[Split]:
    splits: list[Split] = []
    start = train_window
    while start + test_window <= len(index):
        train_idx = index[start - train_window : start]
        test_idx = index[start : start + test_window]
        splits.append(Split(train_idx=train_idx, test_idx=test_idx))
        start += step
    return splits
