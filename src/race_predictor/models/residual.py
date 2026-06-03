"""ML residual corrector trained on time-series holdouts."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from race_predictor.constants import DEFAULT_TEMP_F, DISTANCE_BUCKETS_MI, RACE_DISTANCES_MI
from race_predictor.data.models import Run, TrainedModel
from race_predictor.features.fitness import (
    FEATURE_NAMES,
    compute_fitness_features,
    feature_vector,
    features_to_array,
)
from race_predictor.models.baseline import predict_baseline


def distance_label_for_mi(distance_mi: float) -> str | None:
    for label, (lo, hi) in DISTANCE_BUCKETS_MI.items():
        if lo <= distance_mi <= hi:
            return label
    return None


def _default_temp_f(runs: list[Run]) -> float:
    temps = [run.temp_f for run in runs if run.temp_f is not None]
    return float(np.mean(temps)) if temps else DEFAULT_TEMP_F


def _is_holdout_candidate(run: Run) -> bool:
    if run.is_likely_race:
        return True
    return distance_label_for_mi(run.distance_mi) is not None


def build_training_rows(runs: list[Run]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[dict] = []

    for idx, holdout in enumerate(runs):
        if not _is_holdout_candidate(holdout):
            continue

        label = distance_label_for_mi(holdout.distance_mi)
        if label is None:
            continue

        prior = [run for run in runs if run.date < holdout.date]
        if len(prior) < 3:
            continue

        temp_f = holdout.temp_f if holdout.temp_f is not None else _default_temp_f(prior)
        baseline = predict_baseline(
            prior,
            holdout.date,
            label,
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
        )
        if baseline is None or baseline.predicted_time_sec <= 0:
            continue

        features = compute_fitness_features(prior, holdout.date, label)
        row = feature_vector(
            features,
            RACE_DISTANCES_MI[label],
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
        )
        residual = holdout.moving_time_sec - baseline.predicted_time_sec

        xs.append(features_to_array(FEATURE_NAMES, row))
        ys.append(residual)
        meta.append(
            {
                "date": holdout.date.isoformat(),
                "distance_label": label,
                "actual_sec": holdout.moving_time_sec,
                "baseline_sec": baseline.predicted_time_sec,
            }
        )

    if not xs:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), []

    return np.vstack(xs), np.array(ys), meta


def train_residual_model(runs: list[Run]) -> TrainedModel:
    x, y, _meta = build_training_rows(runs)
    default_temp = _default_temp_f(runs)

    if len(y) < 5:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(np.zeros((1, len(FEATURE_NAMES))), np.array([0.0]))
        return TrainedModel(
            residual_model=model,
            feature_names=FEATURE_NAMES.copy(),
            default_temp_f=default_temp,
        )

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(x, y)

    return TrainedModel(
        residual_model=model,
        feature_names=FEATURE_NAMES.copy(),
        default_temp_f=default_temp,
    )


def predict_residual(
    model: TrainedModel,
    features: dict[str, float],
    target_distance_mi: float,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float,
) -> float:
    row = feature_vector(features, target_distance_mi, elev_gain_ft, elev_loss_ft, temp_f)
    x = features_to_array(model.feature_names, row).reshape(1, -1)
    return float(model.residual_model.predict(x)[0])
