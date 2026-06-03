"""Trailing-window utilities with distance-specific lookback."""

from __future__ import annotations

from datetime import datetime, timedelta

from race_predictor.constants import DEFAULT_DISTANCE_LABEL, WINDOW_DAYS_BY_DISTANCE
from race_predictor.data.models import Run


def window_days_for(distance_label: str) -> int:
    try:
        return WINDOW_DAYS_BY_DISTANCE[distance_label]
    except KeyError as exc:
        valid = ", ".join(WINDOW_DAYS_BY_DISTANCE)
        raise ValueError(f"Unknown distance {distance_label!r}. Expected one of: {valid}") from exc


def runs_in_window(
    runs: list[Run],
    as_of: datetime,
    *,
    distance_label: str | None = None,
    days: int | None = None,
) -> list[Run]:
    """Return runs in [as_of - window, as_of).

    Window length defaults from distance_label (10 weeks for 5K/10K, 12 weeks for Half/Marathon).
    """
    if days is None:
        label = distance_label or DEFAULT_DISTANCE_LABEL
        days = window_days_for(label)

    start = as_of - timedelta(days=days)
    return [run for run in runs if start <= run.date < as_of]
