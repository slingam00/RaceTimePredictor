"""Train, load, and run hybrid baseline + ML residual predictions."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

from race_predictor.constants import DEFAULT_TEMP_F, RACE_DISTANCES_MI
from race_predictor.confidence.scoring import confidence_score, prediction_interval
from race_predictor.data.models import RacePrediction, Run, TrainedModel
from race_predictor.features.fitness import compute_fitness_features
from race_predictor.models.baseline import predict_baseline
from race_predictor.models.population import (
    predict_population_residual,
    train_population_residual_model,
)
from race_predictor.models.residual import predict_residual, train_residual_model

DEFAULT_MODEL_PATH = Path("models/trained_model.pkl")


def train(
    runs: list[Run],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    population: bool = False,
) -> TrainedModel:
    if population:
        model = train_population_residual_model(runs)
    else:
        model = train_residual_model(runs)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return model


def train_population(
    runs: list[Run],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> TrainedModel:
    return train(runs, model_path, population=True)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> TrainedModel:
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    if not getattr(model, "training_mode", None):
        model.training_mode = "athlete"
    return model


def predict_race(
    runs: list[Run],
    model: TrainedModel,
    as_of: datetime,
    distance_label: str,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None = None,
) -> RacePrediction | None:
    if temp_f is None:
        temp_f = model.default_temp_f

    prior = [run for run in runs if run.date < as_of]
    if not prior:
        prior = runs

    baseline = predict_baseline(
        prior, as_of, distance_label, elev_gain_ft, elev_loss_ft, temp_f
    )
    if baseline is None:
        return None

    features = compute_fitness_features(prior, as_of, distance_label)
    if model.training_mode == "population":
        residual = predict_population_residual(
            model,
            prior,
            as_of,
            distance_label,
            elev_gain_ft,
            elev_loss_ft,
            temp_f,
        )
    else:
        residual = predict_residual(
            model,
            features,
            baseline.distance_mi,
        )
    predicted = max(0.0, baseline.predicted_time_sec + residual)
    pace = (predicted / 60.0) / baseline.distance_mi
    low, high = prediction_interval(model, distance_label, predicted, features)
    conf = confidence_score(
        model,
        distance_label,
        features,
        baseline.vdot_time_sec,
        baseline.riegel_time_sec,
        predicted,
    )

    return RacePrediction(
        distance_label=distance_label,
        distance_mi=baseline.distance_mi,
        baseline_time_sec=baseline.predicted_time_sec,
        residual_sec=residual,
        predicted_time_sec=predicted,
        vdot_time_sec=baseline.vdot_time_sec,
        riegel_time_sec=baseline.riegel_time_sec,
        pace_min_per_mi=pace,
        interval_low_sec=low,
        interval_high_sec=high,
        confidence=conf,
    )


def predict_all(
    runs: list[Run],
    model: TrainedModel,
    as_of: datetime,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None = None,
) -> list[RacePrediction]:
    if temp_f is None:
        temp_f = model.default_temp_f

    predictions: list[RacePrediction] = []
    for label in RACE_DISTANCES_MI:
        result = predict_race(
            runs, model, as_of, label, elev_gain_ft, elev_loss_ft, temp_f
        )
        if result is not None:
            predictions.append(result)
    return predictions
