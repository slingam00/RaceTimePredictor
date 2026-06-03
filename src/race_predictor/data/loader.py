"""Strava activities.csv loader with US customary normalization."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from race_predictor.data.models import Run
from race_predictor.units import (
    celsius_to_fahrenheit,
    meters_to_feet,
    meters_to_miles,
    min_per_km_to_min_per_mile,
    pace_min_per_mile,
)

METERS_PER_MILE = 1609.344

RACE_KEYWORDS = re.compile(
    r"\b(race|marathon|half|5k|10k|turkey|trot|parkrun|park run)\b",
    re.IGNORECASE,
)

DATE_FORMATS = (
    "%b %d, %Y, %I:%M:%S %p",
    "%b %d, %Y",
)

MIN_DISTANCE_M = 400
MIN_MOVING_TIME_SEC = 120


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_date(value: str) -> datetime | None:
    text = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _distance_to_meters(raw_distance: float) -> float:
    """Strava duplicate Distance columns: later value is meters; small values may be miles."""
    if raw_distance < 100:
        return raw_distance * METERS_PER_MILE
    return raw_distance


def _parse_gap_pace_min_per_mi(
    gap_value: float | None,
    moving_sec: float,
    distance_mi: float,
) -> float | None:
    derived = pace_min_per_mile(moving_sec, distance_mi)
    if gap_value is None or gap_value <= 0:
        return derived
    # Strava export stores grade-adjusted pace as min/km when under ~15.
    if gap_value < 15:
        return min_per_km_to_min_per_mile(gap_value)
    return derived


def _distance_bucket_label(distance_mi: float) -> str | None:
    if 2.95 <= distance_mi <= 3.25:
        return "5K"
    if 5.9 <= distance_mi <= 6.5:
        return "10K"
    if 12.5 <= distance_mi <= 13.7:
        return "Half"
    if 25.0 <= distance_mi <= 27.5:
        return "Marathon"
    return None


def _is_likely_race(name: str, distance_mi: float) -> bool:
    if RACE_KEYWORDS.search(name):
        return True
    bucket = _distance_bucket_label(distance_mi)
    if bucket is None:
        return False
    lowered = name.lower()
    return any(kw in lowered for kw in ("marathon", "half", "5k", "10k", "race"))


def _parse_temp_f(row: dict[str, str]) -> float | None:
    weather_temp = _parse_float(row.get("Weather Temperature"))
    if weather_temp is not None:
        return celsius_to_fahrenheit(weather_temp)

    avg_temp = _parse_float(row.get("Average Temperature"))
    # Strava body/device temps are often °F already; weather temps are °C.
    if avg_temp is not None and avg_temp > 20:
        return celsius_to_fahrenheit(avg_temp)
    return None


def load_runs(csv_path: str | Path) -> list[Run]:
    """Load and normalize runs from a Strava activities.csv export."""
    path = Path(csv_path)
    runs: list[Run] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Activity Type", "").strip() != "Run":
                continue

            raw_distance = _parse_float(row.get("Distance"))
            moving_sec = _parse_float(row.get("Moving Time"))
            activity_date = _parse_date(row.get("Activity Date", ""))

            if raw_distance is None or moving_sec is None or activity_date is None:
                continue

            distance_m = _distance_to_meters(raw_distance)
            if distance_m < MIN_DISTANCE_M or moving_sec < MIN_MOVING_TIME_SEC:
                continue

            distance_mi = meters_to_miles(distance_m)
            name = row.get("Activity Name", "").strip()

            runs.append(
                Run(
                    activity_id=str(row.get("Activity ID", "")),
                    date=activity_date,
                    name=name,
                    distance_mi=distance_mi,
                    moving_time_sec=moving_sec,
                    elev_gain_ft=meters_to_feet(_parse_float(row.get("Elevation Gain")) or 0.0),
                    elev_loss_ft=meters_to_feet(_parse_float(row.get("Elevation Loss")) or 0.0),
                    gap_pace_min_per_mi=_parse_gap_pace_min_per_mi(
                        _parse_float(row.get("Average Grade Adjusted Pace")),
                        moving_sec,
                        distance_mi,
                    ),
                    avg_hr=_parse_float(row.get("Average Heart Rate")),
                    relative_effort=_parse_float(row.get("Relative Effort")),
                    temp_f=_parse_temp_f(row),
                    is_likely_race=_is_likely_race(name, distance_mi),
                )
            )

    runs.sort(key=lambda run: run.date)
    return runs
