"""Tests for population residual training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from race_predictor.data.benchmark_loader import load_benchmark_corpus
from race_predictor.data.models import Run
from race_predictor.models.population import (
    build_population_training_rows,
    train_population_residual_model,
)
from race_predictor.models.predictor import predict_all, train_population

CORPUS = Path("benchmarks/runsignup_corpus.csv")


def _finisher(
    *,
    athlete_id: str,
    distance_mi: float,
    time_sec: float,
    elev_gain_ft: float = 100,
    elev_loss_ft: float = 100,
    temp_f: float = 60,
) -> Run:
    return Run(
        activity_id=f"{athlete_id}:1",
        date=datetime(2024, 5, 1),
        name="Sample Race",
        distance_mi=distance_mi,
        moving_time_sec=time_sec,
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        gap_pace_min_per_mi=None,
        avg_hr=None,
        relative_effort=None,
        temp_f=temp_f,
        is_likely_race=True,
        athlete_id=athlete_id,
    )


def test_build_population_training_rows_from_synthetic_corpus():
    runs = [
        _finisher(athlete_id=str(i), distance_mi=3.1, time_sec=1200 + i * 20)
        for i in range(80)
    ]
    x, y, meta = build_population_training_rows(runs)
    assert x.shape[0] == len(y) == len(meta)
    assert x.shape[1] == 5
    assert len(y) >= 50


def test_train_population_model_sets_mode(tmp_path):
    runs = [
        _finisher(athlete_id=str(i), distance_mi=3.1, time_sec=1200 + i * 15)
        for i in range(100)
    ]
    model = train_population_residual_model(runs)
    assert model.training_mode == "population"
    assert len(model.feature_names) == 5


@pytest.mark.skipif(not CORPUS.exists(), reason="runsignup corpus not present")
def test_train_population_from_runsignup_corpus(tmp_path):
    runs = load_benchmark_corpus(CORPUS)
    model = train_population(runs, tmp_path / "population_model.pkl")
    assert model.training_mode == "population"
    assert model.residual_stats


@pytest.mark.skipif(not CORPUS.exists(), reason="runsignup corpus not present")
def test_population_model_predicts_with_strava_history(tmp_path):
    from race_predictor.data.loader import load_runs

    strava_csv = Path("data/activities.csv")
    if not strava_csv.exists():
        pytest.skip("Strava export not present")

    corpus_runs = load_benchmark_corpus(CORPUS)
    model = train_population(corpus_runs, tmp_path / "population_model.pkl")
    athlete_runs = load_runs(strava_csv)
    as_of = athlete_runs[-1].date

    predictions = predict_all(
        athlete_runs,
        model,
        as_of,
        elev_gain_ft=200,
        elev_loss_ft=200,
        temp_f=65,
    )
    assert len(predictions) == 4
