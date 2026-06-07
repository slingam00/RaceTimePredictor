"""Tests for VDOT/Riegel baseline predictor."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_predictor.constants import RACE_DISTANCES_MI
from race_predictor.data.loader import load_runs
from race_predictor.data.models import Run
from race_predictor.models.adjustments import apply_course_adjustments, temperature_factor
from race_predictor.models.baseline import baseline_time_sec, predict_all_baselines
from race_predictor.models.riegel import riegel_time
from race_predictor.models.vdot import time_from_vdot, vdot_from_effort

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


def test_vdot_roundtrip():
    vdot = vdot_from_effort(RACE_DISTANCES_MI["5K"], 22 * 60 + 30)
    assert vdot is not None
    t = time_from_vdot(vdot, RACE_DISTANCES_MI["5K"])
    assert t == pytest.approx(22 * 60 + 30, rel=0.05)


def test_riegel_double_distance():
    t5k = 20 * 60.0
    t10k = riegel_time(t5k, RACE_DISTANCES_MI["5K"], RACE_DISTANCES_MI["10K"])
    assert t10k is not None
    assert t10k > t5k


def test_course_adjustments_increase_with_heat_and_elevation():
    base = 3600.0
    flat_cool = apply_course_adjustments(base, 6.0, 0, 0, 50)
    hot_flat = apply_course_adjustments(base, 6.0, 0, 0, 80)
    hot_uphill = apply_course_adjustments(base, 6.0, 500, 0, 80)
    assert hot_flat > flat_cool
    assert hot_uphill > hot_flat
    assert temperature_factor(72) > 1.0


def test_equal_gain_and_loss_is_flat_elevation():
    base = 3600.0
    flat = apply_course_adjustments(base, 6.0, 0, 0, 50)
    rolling = apply_course_adjustments(base, 6.0, 492, 492, 50)
    assert rolling == pytest.approx(flat)


def test_downhill_course_faster_than_flat():
    base = 3600.0
    flat = apply_course_adjustments(base, 26.2, 0, 0, 50)
    downhill = apply_course_adjustments(base, 26.2, 0, 1000, 50)
    assert downhill < flat
    assert flat - downhill == pytest.approx(45.0, rel=1e-3)


def test_baseline_increases_with_distance():
    runs = [_sample_run(i * 3 + 1, 4.0, 8.5) for i in range(25)]
    as_of = datetime(2026, 6, 1)

    t5k, _, _ = baseline_time_sec(runs, as_of, "5K")
    t10k, _, _ = baseline_time_sec(runs, as_of, "10K")
    t_half, _, _ = baseline_time_sec(runs, as_of, "Half")

    assert t5k is not None and t10k is not None and t_half is not None
    assert t5k < t10k < t_half


def test_predict_all_baselines_from_export():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    as_of = runs[-1].date + timedelta(days=1)
    predictions = predict_all_baselines(
        runs, as_of, elev_gain_ft=492, elev_loss_ft=492, temp_f=72
    )

    assert len(predictions) == 4
    labels = [p.distance_label for p in predictions]
    assert labels == ["5K", "10K", "Half", "Marathon"]

    times = [p.predicted_time_sec for p in predictions]
    assert times == sorted(times)
    assert all(p.vdot_time_sec is not None for p in predictions)
    assert all(p.pace_min_per_mi > 0 for p in predictions)
