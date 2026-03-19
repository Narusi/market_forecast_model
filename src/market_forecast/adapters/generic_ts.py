from __future__ import annotations


def generic_feature_flags() -> dict[str, bool]:
    return {
        "technical": False,
        "macro": False,
        "correlation": False,
        "signals": True,
    }
