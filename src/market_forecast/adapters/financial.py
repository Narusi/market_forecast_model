from __future__ import annotations


def financial_feature_flags() -> dict[str, bool]:
    return {
        "technical": True,
        "macro": True,
        "correlation": True,
        "signals": True,
    }
