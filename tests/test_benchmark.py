"""Tests for Phase 2 athlete-level benchmark."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_predictor.data.benchmark_loader import load_benchmark_corpus
from race_predictor.data.models import Run
from race_predictor.evaluate.benchmark import BENCHMARK_VARIANTS, run_benchmark
from race_predictor.evaluate.split import split_by_athlete
from race_predictor.models.baseline import predict_variant_time_sec
from race_predictor.models.ultrassignup import predict_ultrassignup_time_sec, reference_winning_times

SAMPLE_CORPUS = Path("benchmarks/sample_corpus.csv")


def _sample_run(
    athlete_id: str,
    activity_id: str,
    day_offset: int,
    distance_mi: float,
    moving_time_sec: float,
    *,
    is_race: bool = False,
    name: str = "Run",
) -> Run:
    return Run(
        activity_id=activity_id,
        date=datetime(2025, 1, 1) + timedelta(days=day_offset),
        name=name,
        distance_mi=distance_mi,
        moving_time_sec=moving_time_sec,
        elev_gain_ft=30.0,
        elev_loss_ft=30.0,
        gap_pace_min_per_mi=(moving_time_sec / 60.0) / distance_mi,
        avg_hr=150.0,
        relative_effort=70.0,
        temp_f=60.0,
        is_likely_race=is_race,
        athlete_id=athlete_id,
    )


def _synthetic_corpus() -> list[Run]:
    runs: list[Run] = []
    for athlete_idx in range(4):
        athlete = f"a{athlete_idx}"
        for week in range(8):
            runs.append(
                _sample_run(
                    athlete,
                    f"{athlete}-easy-{week}",
                    week * 7,
                    4.5,
                    2700 + athlete_idx * 60,
                )
            )
        runs.append(
            _sample_run(
                athlete,
                f"{athlete}-5k",
                60,
                3.10686,
                1200 + athlete_idx * 90,
                is_race=True,
                name="5K Race",
            )
        )
    return runs


def test_split_by_athlete_has_no_overlap():
    runs = _synthetic_corpus()
    train_runs, test_runs, train_ids, test_ids = split_by_athlete(runs, holdout_frac=0.25, seed=1)

    assert train_ids
    assert test_ids
    assert set(train_ids).isdisjoint(test_ids)
    assert {run.athlete_id for run in train_runs} <= set(train_ids)
    assert {run.athlete_id for run in test_runs} <= set(test_ids)


def test_variant_predictors_differ():
    runs = _synthetic_corpus()[:9]
    as_of = datetime(2025, 3, 1)
    riegel = predict_variant_time_sec(runs, as_of, "5K", 30, 30, 60, variant="riegel")
    vdot = predict_variant_time_sec(runs, as_of, "5K", 30, 30, 60, variant="vdot")
    assert riegel is not None
    assert vdot is not None
    assert riegel != vdot


def test_ultrassignup_predicts_with_reference_times():
    runs = _synthetic_corpus()
    refs = reference_winning_times(runs[:12])
    prior = runs[:8]
    predicted = predict_ultrassignup_time_sec(
        prior,
        datetime(2025, 3, 1),
        "5K",
        30.0,
        30.0,
        60.0,
        refs,
    )
    assert predicted is not None
    assert predicted > 0


def test_run_benchmark_produces_variant_metrics():
    result = run_benchmark(_synthetic_corpus(), holdout_frac=0.25, seed=42)

    assert result.train_athletes >= 1
    assert result.test_athletes >= 1
    assert result.holdout_count >= 1
    assert set(result.metrics_by_variant) == set(BENCHMARK_VARIANTS)
    assert result.generalization["verdict"] in {"generalizes", "mixed", "athlete_specific", "inconclusive"}


@pytest.mark.skipif(not SAMPLE_CORPUS.exists(), reason="sample benchmark corpus not present")
def test_load_sample_corpus():
    runs = load_benchmark_corpus(SAMPLE_CORPUS)
    assert len(runs) >= 100
    assert len({run.athlete_id for run in runs}) >= 4


@pytest.mark.skipif(not SAMPLE_CORPUS.exists(), reason="sample benchmark corpus not present")
def test_sample_corpus_benchmark_runs():
    runs = load_benchmark_corpus(SAMPLE_CORPUS)
    result = run_benchmark(runs, holdout_frac=0.25, seed=42)

    assert result.holdout_count >= 2
    hybrid_5k = next(
        row for row in result.comparison_table if row.variant == "hybrid" and row.distance_label == "5K"
    )
    assert hybrid_5k.count >= 1
    assert hybrid_5k.mape >= 0
