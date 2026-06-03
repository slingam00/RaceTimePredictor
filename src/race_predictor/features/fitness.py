"""Trailing fitness index with distance-specific lookback windows."""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean, pstdev

from race_predictor.data.models import Run
from race_predictor.features.window import runs_in_window, window_days_for
from race_predictor.models.vdot import vdot_from_effort


def _recency_weight(run_date: datetime, as_of: datetime, window_days: int) -> float:
    age_days = (as_of - run_date).days
    if age_days < 0 or age_days >= window_days:
        return 0.0
    return 1.0 - (age_days / window_days)


def weighted_vdot(runs: list[Run], as_of: datetime, window_days: int) -> float | None:
    values: list[tuple[float, float]] = []
    for run in runs:
        vdot = vdot_from_effort(run.distance_mi, run.moving_time_sec)
        if vdot is None or vdot < 20 or vdot > 85:
            continue
        weight = _recency_weight(run.date, as_of, window_days)
        if weight > 0:
            values.append((vdot, weight))
    if not values:
        return None
    total_w = sum(weight for _, weight in values)
    return sum(v * w for v, w in values) / total_w


def _weekly_mileage(runs: list[Run], as_of: datetime, window_days: int) -> list[float]:
    weeks: dict[int, float] = {}
    start = as_of - timedelta(days=window_days)
    for run in runs:
        if run.date < start or run.date >= as_of:
            continue
        week_key = (run.date - start).days // 7
        weeks[week_key] = weeks.get(week_key, 0.0) + run.distance_mi
    if not weeks:
        return []
    return [weeks.get(i, 0.0) for i in range(max(weeks) + 1)]


def compute_fitness_features(
    runs: list[Run],
    as_of: datetime,
    distance_label: str,
) -> dict[str, float]:
    """Compute fitness features using the window for the target race distance."""
    window_days = window_days_for(distance_label)
    window = runs_in_window(runs, as_of, days=window_days)

    empty = {
        "window_days": float(window_days),
        "total_miles": 0.0,
        "run_count": 0.0,
        "runs_per_week": 0.0,
        "longest_run_mi": 0.0,
        "weekly_mileage_std": 0.0,
        "weighted_vdot": 0.0,
        "best_gap_pace": 0.0,
        "avg_relative_effort": 0.0,
        "avg_hr": 0.0,
        "days_since_last_run": float(window_days),
        "long_run_count": 0.0,
    }
    if not window:
        return empty

    total_miles = sum(run.distance_mi for run in window)
    run_count = len(window)
    longest = max(run.distance_mi for run in window)
    weekly = _weekly_mileage(runs, as_of, window_days)
    weekly_std = pstdev(weekly) if len(weekly) > 1 else 0.0
    vdot = weighted_vdot(window, as_of, window_days) or 0.0

    gap_paces = [run.gap_pace_min_per_mi or run.pace_min_per_mi for run in window]
    gap_paces = [pace for pace in gap_paces if pace is not None and 4 <= pace <= 20]
    best_gap = min(gap_paces) if gap_paces else 0.0

    efforts = [run.relative_effort for run in window if run.relative_effort is not None]
    hrs = [run.avg_hr for run in window if run.avg_hr is not None]
    long_runs = sum(1 for run in window if run.distance_mi >= 8.0)

    last_run = max(window, key=lambda run: run.date)
    days_since = (as_of - last_run.date).days

    return {
        "window_days": float(window_days),
        "total_miles": total_miles,
        "run_count": float(run_count),
        "runs_per_week": run_count / (window_days / 7.0),
        "longest_run_mi": longest,
        "weekly_mileage_std": weekly_std,
        "weighted_vdot": vdot,
        "best_gap_pace": best_gap,
        "avg_relative_effort": mean(efforts) if efforts else 0.0,
        "avg_hr": mean(hrs) if hrs else 0.0,
        "days_since_last_run": float(days_since),
        "long_run_count": float(long_runs),
    }
