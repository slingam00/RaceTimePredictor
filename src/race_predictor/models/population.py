"""Population residual model trained from multi-athlete race finisher corpora."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from race_predictor.constants import DEFAULT_TEMP_F, RACE_DISTANCES_MI
from race_predictor.data.models import Run, TrainedModel
from race_predictor.features.fitness import compute_fitness_features, features_to_array
from race_predictor.models.adjustments import apply_course_adjustments
from race_predictor.models.residual import _default_temp_f, _residual_stats, distance_label_for_mi
from race_predictor.models.vdot import time_from_vdot, vdot_from_effort

POPULATION_FEATURE_NAMES = [
    "weighted_vdot",
    "elev_gain_ft",
    "elev_loss_ft",
    "temp_f",
    "target_distance_mi",
]

MIN_POPULATION_ROWS = 5


def build_population_training_rows(
    runs: list[Run],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build residual targets from finisher times vs VDOT + course-adjusted expectation."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[dict] = []

    for run in runs:
        label = distance_label_for_mi(run.distance_mi)
        if label is None:
            continue

        vdot = vdot_from_effort(run.distance_mi, run.moving_time_sec)
        if vdot is None or vdot < 20 or vdot > 85:
            continue

        temp_f = run.temp_f if run.temp_f is not None else DEFAULT_TEMP_F
        flat_time = time_from_vdot(vdot, run.distance_mi)
        if flat_time is None or flat_time <= 0:
            continue

        expected = apply_course_adjustments(
            flat_time,
            run.distance_mi,
            run.elev_gain_ft,
            run.elev_loss_ft,
            temp_f,
        )
        residual = run.moving_time_sec - expected
        row = {
            "weighted_vdot": vdot,
            "elev_gain_ft": run.elev_gain_ft,
            "elev_loss_ft": run.elev_loss_ft,
            "temp_f": temp_f,
            "target_distance_mi": run.distance_mi,
        }

        xs.append(features_to_array(POPULATION_FEATURE_NAMES, row))
        ys.append(residual)
        meta.append(
            {
                "athlete_id": run.athlete_id,
                "date": run.date.isoformat(),
                "distance_label": label,
                "actual_sec": run.moving_time_sec,
                "baseline_sec": expected,
            }
        )

    if not xs:
        return np.empty((0, len(POPULATION_FEATURE_NAMES))), np.empty(0), []

    return np.vstack(xs), np.array(ys), meta


def population_feature_row(
    *,
    weighted_vdot: float,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float,
    target_distance_mi: float,
) -> dict[str, float]:
    return {
        "weighted_vdot": weighted_vdot,
        "elev_gain_ft": elev_gain_ft,
        "elev_loss_ft": elev_loss_ft,
        "temp_f": temp_f,
        "target_distance_mi": target_distance_mi,
    }


def resolve_weighted_vdot(
    runs: list[Run],
    as_of: datetime,
    distance_label: str,
) -> float:
    features = compute_fitness_features(runs, as_of, distance_label)
    vdot = features["weighted_vdot"]
    if vdot > 0:
        return vdot

    vdots = [
        value
        for run in runs
        if run.date < as_of
        and (value := vdot_from_effort(run.distance_mi, run.moving_time_sec)) is not None
        and 20 <= value <= 85
    ]
    if vdots:
        return max(vdots)
    return 0.0


def predict_population_residual(
    model: TrainedModel,
    runs: list[Run],
    as_of: datetime,
    distance_label: str,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float,
) -> float:
    target_distance_mi = RACE_DISTANCES_MI[distance_label]
    vdot = resolve_weighted_vdot(runs, as_of, distance_label)
    row = population_feature_row(
        weighted_vdot=vdot,
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        temp_f=temp_f,
        target_distance_mi=target_distance_mi,
    )
    x = features_to_array(model.feature_names, row).reshape(1, -1)
    return float(model.residual_model.predict(x)[0])


def train_population_residual_model(runs: list[Run]) -> TrainedModel:
    x, y, meta = build_population_training_rows(runs)
    default_temp = _default_temp_f(runs)

    if len(y) < MIN_POPULATION_ROWS:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(np.zeros((1, len(POPULATION_FEATURE_NAMES))), np.array([0.0]))
        return TrainedModel(
            residual_model=model,
            feature_names=POPULATION_FEATURE_NAMES.copy(),
            default_temp_f=default_temp,
            residual_stats={},
            training_mode="population",
        )

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(x, y)
    preds = model.predict(x)
    stats = _residual_stats(meta, preds)

    return TrainedModel(
        residual_model=model,
        feature_names=POPULATION_FEATURE_NAMES.copy(),
        default_temp_f=default_temp,
        residual_stats=stats,
        training_mode="population",
    )
