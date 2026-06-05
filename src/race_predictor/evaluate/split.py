"""Athlete-level train/test splits for benchmark evaluation."""

from __future__ import annotations

import random

from race_predictor.data.models import Run


def split_by_athlete(
    runs: list[Run],
    holdout_frac: float = 0.25,
    seed: int = 42,
) -> tuple[list[Run], list[Run], list[str], list[str]]:
    """Hold out a fraction of athletes entirely for testing.

    Returns (train_runs, test_runs, train_athlete_ids, test_athlete_ids).
    """
    if not runs:
        return [], [], [], []
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be between 0 and 1.")

    athletes = sorted({run.athlete_id for run in runs})
    if len(athletes) < 2:
        raise ValueError("Benchmark corpus must include at least two athletes.")

    rng = random.Random(seed)
    shuffled = athletes.copy()
    rng.shuffle(shuffled)

    holdout_count = max(1, round(len(shuffled) * holdout_frac))
    if holdout_count >= len(shuffled):
        holdout_count = len(shuffled) - 1

    test_ids = set(shuffled[:holdout_count])
    train_ids = set(shuffled[holdout_count:])

    train_runs = [run for run in runs if run.athlete_id in train_ids]
    test_runs = [run for run in runs if run.athlete_id in test_ids]
    return train_runs, test_runs, sorted(train_ids), sorted(test_ids)
