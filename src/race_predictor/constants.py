"""Shared constants (US customary)."""

from __future__ import annotations

# Trailing windows: 10 weeks for short races, 12 weeks for long races.
WINDOW_DAYS_BY_DISTANCE: dict[str, int] = {
    "5K": 70,
    "10K": 70,
    "Half": 84,
    "Marathon": 84,
}

RACE_DISTANCES_MI: dict[str, float] = {
    "5K": 3.10686,
    "10K": 6.21371,
    "Half": 13.1094,
    "Marathon": 26.2188,
}

DEFAULT_DISTANCE_LABEL = "5K"

RIEGEL_EXPONENT = 1.06
VDOT_BASELINE_WEIGHT = 0.6
RIEGEL_BASELINE_WEIGHT = 0.4

OPTIMAL_TEMP_F = 52.0
DEFAULT_TEMP_F = 60.0
