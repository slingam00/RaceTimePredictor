"""Hybrid VDOT + Riegel baseline race time predictions."""

from __future__ import annotations

from datetime import datetime

from race_predictor.constants import (
    DEFAULT_TEMP_F,
    RACE_DISTANCES_MI,
    RIEGEL_BASELINE_WEIGHT,
    VDOT_BASELINE_WEIGHT,
)
from race_predictor.data.models import BaselinePrediction, Run
from race_predictor.features.fitness import compute_fitness_features
from race_predictor.features.window import runs_in_window
from race_predictor.models.adjustments import apply_course_adjustments
from race_predictor.models.riegel import riegel_time
from race_predictor.models.vdot import time_from_vdot, vdot_from_effort


def best_effort_for_distance(
    runs: list[Run],
    as_of: datetime,
    target_distance_mi: float,
    distance_label: str,
) -> Run | None:
    """Best (fastest pace) run within 50–150% of target distance in the training window."""
    window = runs_in_window(runs, as_of, distance_label=distance_label)
    candidates = [
        run
        for run in window
        if 0.5 * target_distance_mi <= run.distance_mi <= 1.5 * target_distance_mi
    ]
    if not candidates:
        candidates = window
    if not candidates:
        return None
    return min(candidates, key=lambda run: run.moving_time_sec / run.distance_mi)


def _vdot_baseline_time(
    runs: list[Run],
    as_of: datetime,
    target_distance_mi: float,
    distance_label: str,
) -> float | None:
    features = compute_fitness_features(runs, as_of, distance_label)
    vdot = features["weighted_vdot"]
    if vdot <= 0:
        window = runs_in_window(runs, as_of, distance_label=distance_label)
        vdots = [
            v
            for run in window
            if (v := vdot_from_effort(run.distance_mi, run.moving_time_sec)) is not None
        ]
        vdot = max(vdots) if vdots else 0.0
    if vdot <= 0:
        return None
    return time_from_vdot(vdot, target_distance_mi)


def _riegel_baseline_time(
    runs: list[Run],
    as_of: datetime,
    target_distance_mi: float,
    distance_label: str,
) -> float | None:
    effort = best_effort_for_distance(runs, as_of, target_distance_mi, distance_label)
    if effort is None:
        return None
    return riegel_time(effort.moving_time_sec, effort.distance_mi, target_distance_mi)


def baseline_time_sec(
    runs: list[Run],
    as_of: datetime,
    distance_label: str,
) -> tuple[float | None, float | None, float | None]:
    """Return blended, VDOT-only, and Riegel-only baseline times in seconds."""
    target_distance_mi = RACE_DISTANCES_MI[distance_label]
    vdot_time = _vdot_baseline_time(runs, as_of, target_distance_mi, distance_label)
    riegel_time_sec = _riegel_baseline_time(runs, as_of, target_distance_mi, distance_label)

    if vdot_time is None and riegel_time_sec is None:
        return None, None, None
    if vdot_time is None:
        return riegel_time_sec, None, riegel_time_sec
    if riegel_time_sec is None:
        return vdot_time, vdot_time, None

    blended = VDOT_BASELINE_WEIGHT * vdot_time + RIEGEL_BASELINE_WEIGHT * riegel_time_sec
    return blended, vdot_time, riegel_time_sec


def predict_baseline(
    runs: list[Run],
    as_of: datetime,
    distance_label: str,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None = None,
) -> BaselinePrediction | None:
    """Predict race time using VDOT/Riegel blend plus course adjustments."""
    if temp_f is None:
        temps = [run.temp_f for run in runs if run.temp_f is not None]
        temp_f = sum(temps) / len(temps) if temps else DEFAULT_TEMP_F

    target_distance_mi = RACE_DISTANCES_MI[distance_label]
    blended, vdot_time, riegel_time_sec = baseline_time_sec(runs, as_of, distance_label)
    if blended is None:
        return None

    adjusted = apply_course_adjustments(
        blended, target_distance_mi, elev_gain_ft, elev_loss_ft, temp_f
    )
    pace = (adjusted / 60.0) / target_distance_mi

    return BaselinePrediction(
        distance_label=distance_label,
        distance_mi=target_distance_mi,
        predicted_time_sec=adjusted,
        vdot_time_sec=vdot_time,
        riegel_time_sec=riegel_time_sec,
        blended_time_sec=blended,
        adjusted_time_sec=adjusted,
        pace_min_per_mi=pace,
    )


def predict_all_baselines(
    runs: list[Run],
    as_of: datetime,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None = None,
) -> list[BaselinePrediction]:
    predictions: list[BaselinePrediction] = []
    for label in RACE_DISTANCES_MI:
        result = predict_baseline(
            runs, as_of, label, elev_gain_ft, elev_loss_ft, temp_f
        )
        if result is not None:
            predictions.append(result)
    return predictions
