"""Data models for parsed Strava runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Run:
    activity_id: str
    date: datetime
    name: str
    distance_mi: float
    moving_time_sec: float
    elev_gain_ft: float
    elev_loss_ft: float
    gap_pace_min_per_mi: float | None
    avg_hr: float | None
    relative_effort: float | None
    temp_f: float | None
    is_likely_race: bool = False
    athlete_id: str = "default"

    @property
    def pace_min_per_mi(self) -> float | None:
        if self.distance_mi <= 0 or self.moving_time_sec <= 0:
            return None
        return (self.moving_time_sec / 60.0) / self.distance_mi


@dataclass(frozen=True)
class BaselinePrediction:
    distance_label: str
    distance_mi: float
    predicted_time_sec: float
    vdot_time_sec: float | None
    riegel_time_sec: float | None
    blended_time_sec: float
    adjusted_time_sec: float
    pace_min_per_mi: float


@dataclass
class TrainedModel:
    residual_model: object
    feature_names: list[str]
    default_temp_f: float
    residual_stats: dict[str, dict[str, float]] | None = None
    training_mode: str = "athlete"

    def __post_init__(self) -> None:
        if self.residual_stats is None:
            self.residual_stats = {}
        if not getattr(self, "training_mode", None):
            self.training_mode = "athlete"


@dataclass(frozen=True)
class RacePrediction:
    distance_label: str
    distance_mi: float
    baseline_time_sec: float
    residual_sec: float
    predicted_time_sec: float
    vdot_time_sec: float | None
    riegel_time_sec: float | None
    pace_min_per_mi: float
    interval_low_sec: float
    interval_high_sec: float
    confidence: int
