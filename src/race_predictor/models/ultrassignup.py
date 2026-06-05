"""UltraSignup-style race time predictor (rank vs reference winning time)."""

from __future__ import annotations

from datetime import datetime
from statistics import mean

from race_predictor.constants import RACE_DISTANCES_MI
from race_predictor.data.models import Run
from race_predictor.models.adjustments import apply_course_adjustments
from race_predictor.models.residual import distance_label_for_mi

DEFAULT_RUNNER_RANK = 0.75


def reference_winning_times(train_runs: list[Run]) -> dict[str, float]:
    """Fastest race effort per distance in the training corpus (winner proxy)."""
    refs: dict[str, float] = {}
    for run in train_runs:
        label = distance_label_for_mi(run.distance_mi)
        if label is None:
            continue
        current = refs.get(label)
        if current is None or run.moving_time_sec < current:
            refs[label] = run.moving_time_sec
    return refs


def runner_rank(prior_runs: list[Run], reference_times: dict[str, float]) -> float:
    """Average past race performance vs reference winning times (UltraSignup rank)."""
    scores: list[float] = []
    for run in prior_runs:
        label = distance_label_for_mi(run.distance_mi)
        if label is None:
            continue
        if not (run.is_likely_race or label in reference_times):
            continue
        reference = reference_times.get(label)
        if reference is None or run.moving_time_sec <= 0:
            continue
        scores.append(min(1.0, reference / run.moving_time_sec))

    if not scores:
        return DEFAULT_RUNNER_RANK
    return max(0.05, min(1.0, mean(scores)))


def predict_ultrassignup_time_sec(
    prior_runs: list[Run],
    as_of: datetime,
    distance_label: str,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float,
    reference_times: dict[str, float],
) -> float | None:
    """Predict race time using reference winning time / runner rank."""
    reference = reference_times.get(distance_label)
    if reference is None or reference <= 0:
        return None

    rank = runner_rank(
        [run for run in prior_runs if run.date < as_of],
        reference_times,
    )
    base_time = reference / rank
    target_distance_mi = RACE_DISTANCES_MI[distance_label]
    return apply_course_adjustments(
        base_time, target_distance_mi, elev_gain_ft, elev_loss_ft, temp_f
    )
