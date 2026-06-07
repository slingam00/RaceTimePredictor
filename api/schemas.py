"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    elev_gain_ft: float = Field(ge=0)
    elev_loss_ft: float = Field(ge=0)
    temp_f: Optional[float] = Field(default=None, ge=-20, le=120)
    as_of: Optional[date] = None


class PredictionItem(BaseModel):
    distance_label: str
    predicted_time_sec: float
    pace_min_per_mi: float


class PredictResponse(BaseModel):
    as_of: date
    elev_gain_ft: float
    elev_loss_ft: float
    temp_f: float
    temp_source: str
    predictions: list[PredictionItem]
