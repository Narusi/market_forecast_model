from __future__ import annotations


def forecast_to_signal(value: float, buy_threshold: float, sell_threshold: float) -> str:
    if value >= buy_threshold:
        return "buy"
    if value <= sell_threshold:
        return "sell"
    return "hold"


def vote_to_signal(vote: int) -> str:
    if vote > 0:
        return "buy"
    if vote < 0:
        return "sell"
    return "hold"
