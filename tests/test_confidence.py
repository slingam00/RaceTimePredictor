"""Tests for confidence scoring and backtest evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_predictor.confidence.scoring import (
    confidence_score,
    extrapolation_penalty,
    prediction_interval,
)
from race_predictor.data.loader import load_runs
from race_predictor.data.models import Run, TrainedModel
from race_predictor.evaluate.backtest import backtest_to_dict, run_backtest
from race_predictor.features.fitness import compute_fitness_features
from race_predictor.models.predictor import predict_all, train

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


def test_extrapolation_penalty_increases_with_distance():
    assert extrapolation_penalty(3.1, 8.0) == 0.0
    assert extrapolation_penalty(26.2, 8.0) > extrapolation_penalty(13.1, 8.0)


def test_prediction_interval_width():
    model = TrainedModel(
        residual_model=None,
        feature_names=[],
        default_temp_f=60.0,
        residual_stats={"5K": {"mape": 0.05, "count": 10.0, "p80": 0.08}},
    )
    features = {
        "run_count": 20,
        "total_miles": 80,
        "days_since_last_run": 2,
        "longest_run_mi": 10,
    }
    low, high = prediction_interval(model, "5K", 1800, features)
    assert low < 1800 < high


def test_confidence_score_bounded():
    model = TrainedModel(
        residual_model=None,
        feature_names=[],
        default_temp_f=60.0,
        residual_stats={"5K": {"mape": 0.05, "count": 20.0, "p80": 0.08}},
    )
    features = compute_fitness_features(
        [_sample_run(i * 2 + 1, 4.0, 8.5) for i in range(20)],
        datetime(2026, 6, 1),
        "5K",
    )
    score = confidence_score(model, "5K", features, 1700, 1750, 1800)
    assert 0 <= score <= 100


def test_run_backtest_from_export():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    result = run_backtest(runs)

    assert result.holdout_count > 0
    assert result.training_rows > 0
    assert len(result.metrics) == 4

    metrics_5k = next(m for m in result.metrics if m.label == "5K")
    assert metrics_5k.count > 0
    assert metrics_5k.mape > 0
    assert metrics_5k.interval_coverage_95 is not None

    payload = backtest_to_dict(result)
    assert "metrics" in payload
    assert "holdouts" in payload


def test_predictions_include_confidence_and_intervals(tmp_path):
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    model = train(runs, tmp_path / "model.pkl")
    predictions = predict_all(
        runs, model, runs[-1].date, elev_gain_ft=492, elev_loss_ft=492, temp_f=72
    )

    for p in predictions:
        assert 0 <= p.confidence <= 100
        assert p.interval_low_sec <= p.predicted_time_sec <= p.interval_high_sec

    assert predictions[-1].confidence <= predictions[0].confidence
