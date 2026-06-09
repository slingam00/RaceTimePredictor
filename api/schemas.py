"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class RaceEventSummary(BaseModel):
    event_id: int
    name: str
    distance_label: Optional[str] = None


class RaceSummary(BaseModel):
    race_id: int
    name: str
    city: Optional[str] = None
    state: Optional[str] = None
    next_date: Optional[str] = None
    offered_distances: list[str] = Field(default_factory=list)


class RaceSearchResponse(BaseModel):
    races: list[RaceSummary]
    page: int
    results_per_page: int


class RaceEventDetail(BaseModel):
    event_id: int
    name: str
    distance_label: str
    distance_mi: float


class RaceDetail(BaseModel):
    race_id: int
    name: str
    city: Optional[str] = None
    state: Optional[str] = None
    race_date: date
    elev_gain_ft: Optional[float] = None
    elev_loss_ft: Optional[float] = None
    elev_source: Optional[str] = None
    temp_f: Optional[float] = None
    weather_source: Optional[str] = None
    offered_events: list[RaceEventDetail] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    race_id: Optional[int] = None
    event_id: Optional[int] = None
    elev_gain_ft: Optional[float] = Field(default=None, ge=0)
    elev_loss_ft: Optional[float] = Field(default=None, ge=0)
    temp_f: Optional[float] = Field(default=None, ge=-20, le=120)
    as_of: Optional[date] = None

    @model_validator(mode="after")
    def validate_manual_conditions(self) -> "PredictRequest":
        if self.race_id is None:
            if self.elev_gain_ft is None or self.elev_loss_ft is None:
                raise ValueError(
                    "elev_gain_ft and elev_loss_ft are required when race_id is omitted"
                )
        return self


class PredictionItem(BaseModel):
    distance_label: str
    predicted_time_sec: float
    pace_min_per_mi: float
    interval_low_sec: float
    interval_high_sec: float
    confidence: int


class PredictResponse(BaseModel):
    as_of: date
    elev_gain_ft: float
    elev_loss_ft: float
    temp_f: float
    temp_source: str
    race_id: Optional[int] = None
    race_name: Optional[str] = None
    elev_source: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    predictions: list[PredictionItem]
