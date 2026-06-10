"""Shared helpers for prediction horizon from Strava activities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from api.config import Settings
from race_predictor.data.loader import load_runs
from race_predictor.features.prediction_horizon import (
    format_prediction_horizon_message,
    max_prediction_date,
)


@dataclass(frozen=True)
class PredictionHorizon:
    max_prediction_date: date | None
    message: str | None


def load_prediction_horizon(settings: Settings) -> PredictionHorizon:
    csv_path = settings.data_dir / "activities.csv"
    if not csv_path.is_file():
        return PredictionHorizon(max_prediction_date=None, message=None)

    runs = load_runs(csv_path)
    horizon = max_prediction_date(runs)
    if horizon is None:
        return PredictionHorizon(max_prediction_date=None, message=None)

    return PredictionHorizon(
        max_prediction_date=horizon,
        message=format_prediction_horizon_message(horizon),
    )
