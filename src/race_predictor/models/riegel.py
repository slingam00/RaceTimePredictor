"""Riegel race time extrapolation."""

from __future__ import annotations

from race_predictor.constants import RIEGEL_EXPONENT


def riegel_time(
    known_time_sec: float,
    known_distance_mi: float,
    target_distance_mi: float,
    exponent: float = RIEGEL_EXPONENT,
) -> float | None:
    if known_time_sec <= 0 or known_distance_mi <= 0 or target_distance_mi <= 0:
        return None
    ratio = target_distance_mi / known_distance_mi
    return known_time_sec * (ratio**exponent)
