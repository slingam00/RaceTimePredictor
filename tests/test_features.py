"""Tests for distance-specific trailing windows and fitness features."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_predictor.constants import WINDOW_DAYS_BY_DISTANCE
from race_predictor.data.loader import load_runs
from race_predictor.data.models import Run
from race_predictor.features.fitness import compute_fitness_features
from race_predictor.features.window import runs_in_window, window_days_for

DATA_CSV = Path("data/activities.csv")


def _sample_run(days_ago: int, dist_mi: float, pace_min: float) -> Run:
    as_of = datetime(2026, 6, 1)
    return Run(
        activity_id=str(days_ago),
        date=as_of - timedelta(days=days_ago),
        name="Training Run",
        distance_mi=dist_mi,
        moving_time_sec=pace_min * dist_mi * 60,
        elev_gain_ft=50,
        elev_loss_ft=50,
        gap_pace_min_per_mi=pace_min,
        avg_hr=150,
        relative_effort=50,
        temp_f=60,
    )


def test_window_days_by_distance():
    assert window_days_for("5K") == 70
    assert window_days_for("10K") == 70
    assert window_days_for("Half") == 84
    assert window_days_for("Marathon") == 84
    assert WINDOW_DAYS_BY_DISTANCE["5K"] == 10 * 7
    assert WINDOW_DAYS_BY_DISTANCE["Marathon"] == 12 * 7


def test_runs_in_window_respects_distance_label():
    runs = [_sample_run(i * 7, 4.0, 9.0) for i in range(1, 15)]
    as_of = datetime(2026, 6, 1)

    short_window = runs_in_window(runs, as_of, distance_label="5K")
    long_window = runs_in_window(runs, as_of, distance_label="Marathon")

    assert len(short_window) == 10  # 7–70 days ago, weekly (inclusive at window start)
    assert len(long_window) == 12   # 7–84 days ago, weekly
    assert len(long_window) > len(short_window)


def test_fitness_features_include_window_days():
    runs = [_sample_run(i * 2 + 1, 4.0, 8.5 + i * 0.02) for i in range(30)]
    as_of = datetime(2026, 6, 1)

    short = compute_fitness_features(runs, as_of, "5K")
    long = compute_fitness_features(runs, as_of, "Marathon")

    assert short["window_days"] == 70
    assert long["window_days"] == 84
    assert short["run_count"] <= long["run_count"]
    assert short["total_miles"] <= long["total_miles"]


def test_fitness_features_from_export():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    as_of = runs[-1].date + timedelta(days=1)

    features_5k = compute_fitness_features(runs, as_of, "5K")
    features_m = compute_fitness_features(runs, as_of, "Marathon")

    assert features_5k["weighted_vdot"] > 0
    assert features_m["longest_run_mi"] >= features_5k["longest_run_mi"]
    assert features_m["total_miles"] >= features_5k["total_miles"]
