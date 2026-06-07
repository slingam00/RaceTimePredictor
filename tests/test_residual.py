"""Tests for ML residual corrector and hybrid predictor."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_predictor.data.loader import load_runs
from race_predictor.data.models import Run
from race_predictor.models.predictor import predict_all, train
from race_predictor.models.residual import build_training_rows, train_residual_model

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


def test_build_training_rows_from_synthetic():
    runs = [_sample_run(i * 2 + 1, 3.1, 9.0 + i * 0.05) for i in range(40)]
    x, y, meta = build_training_rows(runs)
    assert x.shape[1] > 0
    assert len(y) == len(meta)
    assert len(y) >= 10


def test_train_residual_model_returns_trained_model():
    runs = [_sample_run(i * 2 + 1, 3.1, 9.0) for i in range(40)]
    model = train_residual_model(runs)
    assert model.feature_names
    assert model.default_temp_f > 0


def test_downhill_race_faster_than_flat_on_export(tmp_path):
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    model = train(runs, tmp_path / "model.pkl")
    as_of = runs[-1].date + timedelta(days=1)

    flat = predict_all(runs, model, as_of, elev_gain_ft=0, elev_loss_ft=0, temp_f=72)
    downhill = predict_all(runs, model, as_of, elev_gain_ft=0, elev_loss_ft=1000, temp_f=72)
    uphill = predict_all(runs, model, as_of, elev_gain_ft=1000, elev_loss_ft=0, temp_f=72)

    for dist in ("5K", "10K", "Half", "Marathon"):
        flat_t = next(p for p in flat if p.distance_label == dist).predicted_time_sec
        down_t = next(p for p in downhill if p.distance_label == dist).predicted_time_sec
        up_t = next(p for p in uphill if p.distance_label == dist).predicted_time_sec
        assert down_t < flat_t < up_t, dist


def test_hybrid_predictions_from_export(tmp_path):
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    model_path = tmp_path / "model.pkl"
    model = train(runs, model_path)

    assert model_path.exists()

    as_of = runs[-1].date + timedelta(days=1)
    predictions = predict_all(
        runs, model, as_of, elev_gain_ft=492, elev_loss_ft=492, temp_f=72
    )

    assert len(predictions) == 4
    times = [p.predicted_time_sec for p in predictions]
    assert times == sorted(times)
    assert all(abs(p.residual_sec) < 3600 for p in predictions)
