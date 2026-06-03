"""VDOT from race efforts (Daniels-Gilbert)."""

from __future__ import annotations

import math

METERS_PER_MILE = 1609.344


def _vo2_from_velocity(velocity_m_min: float) -> float:
    return -4.60 + 0.182258 * velocity_m_min + 0.000104 * velocity_m_min**2


def _percent_max_from_minutes(time_min: float) -> float:
    return (
        0.8
        + 0.1894393 * math.exp(-0.012778 * time_min)
        + 0.2989558 * math.exp(-0.1932605 * time_min)
    )


def vdot_from_effort(distance_mi: float, time_sec: float) -> float | None:
    if distance_mi <= 0 or time_sec <= 0:
        return None
    time_min = time_sec / 60.0
    velocity_m_min = (distance_mi * METERS_PER_MILE) / time_min
    vo2 = _vo2_from_velocity(velocity_m_min)
    pct = _percent_max_from_minutes(time_min)
    if pct <= 0:
        return None
    return vo2 / pct


def time_from_vdot(vdot: float, distance_mi: float) -> float | None:
    """Predict race time (seconds) for a distance at a given VDOT."""
    if vdot <= 0 or distance_mi <= 0:
        return None
    low, high = 60.0, distance_mi * 20.0 * 60.0
    for _ in range(80):
        mid = (low + high) / 2.0
        computed = vdot_from_effort(distance_mi, mid)
        if computed is None:
            return None
        if computed > vdot:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
