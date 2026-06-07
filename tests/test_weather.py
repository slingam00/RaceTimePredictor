"""Tests for Open-Meteo race-day weather."""

from __future__ import annotations

from datetime import date

import pytest

from race_predictor.data.weather import fetch_race_day_weather


def _mock_forecast_payload(max_f: float, min_f: float) -> dict:
    return {
        "daily": {
            "temperature_2m_max": [max_f],
            "temperature_2m_min": [min_f],
        }
    }


def test_forecast_uses_daily_mean_for_near_future():
    today = date(2026, 6, 1)
    race_date = date(2026, 6, 10)

    def loader(url: str) -> dict:
        assert "forecast" in url
        return _mock_forecast_payload(80.0, 60.0)

    result = fetch_race_day_weather(
        42.36,
        -71.06,
        race_date,
        today=today,
        fetch_json=loader,
    )
    assert result.temp_f == pytest.approx(70.0)
    assert result.source == "forecast"


def test_archive_for_past_race_date():
    today = date(2026, 6, 1)
    race_date = date(2025, 4, 20)

    def loader(url: str) -> dict:
        assert "archive" in url
        return _mock_forecast_payload(55.0, 45.0)

    result = fetch_race_day_weather(
        42.36,
        -71.06,
        race_date,
        today=today,
        fetch_json=loader,
    )
    assert result.temp_f == pytest.approx(50.0)
    assert result.source == "archive"


def test_climatology_for_far_future_race():
    today = date(2026, 6, 1)
    race_date = date(2027, 4, 20)
    calls: list[str] = []

    def loader(url: str) -> dict:
        calls.append(url)
        assert "archive" in url
        return _mock_forecast_payload(50.0, 40.0)

    result = fetch_race_day_weather(
        42.36,
        -71.06,
        race_date,
        today=today,
        fetch_json=loader,
    )
    assert result.temp_f == pytest.approx(45.0)
    assert result.source == "typical"
    assert len(calls) == 10


def test_api_failure_returns_model_default():
    today = date(2026, 6, 1)
    race_date = date(2026, 6, 5)

    def loader(url: str) -> dict:
        raise OSError("network down")

    result = fetch_race_day_weather(
        42.36,
        -71.06,
        race_date,
        today=today,
        default_temp_f=62.0,
        fetch_json=loader,
    )
    assert result.temp_f == 62.0
    assert result.source == "model_default"
