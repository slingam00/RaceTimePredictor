"""Prediction date limits based on trailing fitness lookback windows."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from race_predictor.constants import WINDOW_DAYS_BY_DISTANCE
from race_predictor.data.models import Run

MAX_LOOKBACK_DAYS = max(WINDOW_DAYS_BY_DISTANCE.values())


def last_run_date(runs: list[Run]) -> date | None:
    if not runs:
        return None
    return max(run.date for run in runs).date()


def max_prediction_date(runs: list[Run]) -> date | None:
    """Latest race date for which recent training still falls in the lookback window."""
    last = last_run_date(runs)
    if last is None:
        return None
    return last + timedelta(days=MAX_LOOKBACK_DAYS)


def format_prediction_horizon_message(max_date: date) -> str:
    return f"Predictions can only be made up until {max_date:%Y-%m-%d}."


def is_within_prediction_horizon(as_of: date | datetime, runs: list[Run]) -> bool:
    max_date = max_prediction_date(runs)
    if max_date is None:
        return True
    target = as_of.date() if isinstance(as_of, datetime) else as_of
    return target <= max_date
