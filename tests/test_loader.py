"""Tests for Strava CSV loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from race_predictor.data.loader import load_runs
from race_predictor.units import (
    celsius_to_fahrenheit,
    meters_to_feet,
    meters_to_miles,
    min_per_km_to_min_per_mile,
)

DATA_CSV = Path("data/activities.csv")


def test_unit_conversions():
    assert meters_to_miles(1609.344) == pytest.approx(1.0, rel=1e-4)
    assert meters_to_feet(0.3048) == pytest.approx(1.0, rel=1e-4)
    assert celsius_to_fahrenheit(0) == pytest.approx(32.0)
    assert celsius_to_fahrenheit(100) == pytest.approx(212.0)
    assert min_per_km_to_min_per_mile(6.0) == pytest.approx(9.656, rel=1e-3)


def test_load_runs_from_export():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    assert len(runs) == 281
    assert all(run.distance_mi > 0 for run in runs)
    assert all(run.moving_time_sec >= 120 for run in runs)
    assert runs == sorted(runs, key=lambda run: run.date)


def test_run_pace_is_min_per_mile():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    sample = next(run for run in runs if run.activity_id == "18716792497")

    assert sample.distance_mi == pytest.approx(1.559, rel=1e-2)
    assert sample.moving_time_sec == pytest.approx(1047.0)
    assert sample.pace_min_per_mi == pytest.approx(11.18, rel=1e-2)
    assert sample.elev_gain_ft == pytest.approx(72.2, rel=1e-1)


def test_likely_races_flagged():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    race_names = [run.name for run in runs if run.is_likely_race]
    assert any("Marathon" in name for name in race_names)
    assert any("5K" in name or "5k" in name for name in race_names)


def test_elevation_available_for_most_runs():
    if not DATA_CSV.exists():
        pytest.skip("Strava export not present")

    runs = load_runs(DATA_CSV)
    with_elev = sum(1 for run in runs if run.elev_gain_ft > 0)
    assert with_elev >= 250
