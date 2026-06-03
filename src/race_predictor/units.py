"""Unit conversions from Strava metric export to US customary."""

from __future__ import annotations

METERS_PER_MILE = 1609.344
METERS_PER_FOOT = 0.3048
KILOMETERS_PER_MILE = METERS_PER_MILE / 1000.0


def meters_to_miles(meters: float) -> float:
    return meters / METERS_PER_MILE


def meters_to_feet(meters: float) -> float:
    return meters / METERS_PER_FOOT


def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def min_per_km_to_min_per_mile(min_per_km: float) -> float:
    return min_per_km * KILOMETERS_PER_MILE


def pace_min_per_mile(time_sec: float, distance_mi: float) -> float | None:
    if distance_mi <= 0 or time_sec <= 0:
        return None
    return (time_sec / 60.0) / distance_mi
