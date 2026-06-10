"""Tests for FastAPI routes."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.config import Settings, get_settings
from api.main import app
from race_predictor.data.models import RacePrediction
from race_predictor.data.race_enrichment import EnrichedRace, EnrichedRaceEvent
from race_predictor.data.runsignup_client import (
    RunSignupEvent,
    RunSignupRaceSummary,
    RunSignupSearchResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _settings(tmp_path: Path, *, data_dir: Path | None = None) -> Settings:
    resolved_data_dir = data_dir or (tmp_path / "data")
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"placeholder")
    return Settings(
        data_dir=resolved_data_dir,
        model_path=model_path,
        cors_origins=["http://localhost:3000"],
        overrides_path=tmp_path / "overrides.json",
        enrichment_cache_dir=tmp_path / "cache",
        gpx_dir=tmp_path / "gpx",
    )


@pytest.fixture
def client(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "activities.csv").write_text(
        "Activity Type,Activity Date,Distance,Moving Time,Activity Name\n"
        "Run,\"Jun 1, 2026\",5000,1200,Morning Run\n",
        encoding="utf-8",
    )

    def override_settings() -> Settings:
        return _settings(tmp_path)

    app.dependency_overrides[get_settings] = override_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.routes.predict.predict_all")
@patch("api.routes.predict.load_model")
def test_predict_manual_conditions(mock_load_model, mock_predict_all, client):
    mock_load_model.return_value = type("Model", (), {"default_temp_f": 60.0})()
    mock_predict_all.return_value = [
        RacePrediction(
            distance_label="5K",
            distance_mi=3.1,
            baseline_time_sec=1800,
            residual_sec=0,
            predicted_time_sec=1800,
            vdot_time_sec=1800,
            riegel_time_sec=1800,
            pace_min_per_mi=9.6,
            interval_low_sec=1750,
            interval_high_sec=1850,
            confidence=80,
        )
    ]

    response = client.post(
        "/api/predict",
        json={
            "elev_gain_ft": 492,
            "elev_loss_ft": 492,
            "temp_f": 72,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["elev_gain_ft"] == 492
    assert payload["temp_f"] == 72
    assert payload["temp_source"] == "request"
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["distance_label"] == "5K"
    assert payload["predictions"][0]["confidence"] == 80
    assert payload["predictions"][0]["interval_low_sec"] == 1750


@patch("api.routes.races.RunSignupClient.search_races")
def test_search_races(mock_search, client):
    mock_search.return_value = RunSignupSearchResult(
        races=[
            RunSignupRaceSummary(
                race_id=146508,
                name="Bridge to Brew",
                city="Port Huron",
                state="MI",
                next_date="08/09/2026",
                last_date="08/10/2025",
                events=[
                    RunSignupEvent(1140658, "Half Marathon", 13.1, "M", "07:00:00"),
                    RunSignupEvent(1140659, "5K", 3.1, "M", "07:30:00"),
                ],
            )
        ],
        page=1,
        results_per_page=25,
    )

    response = client.get("/api/races/search", params={"q": "bridge"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 1
    assert len(payload["races"]) == 1
    assert payload["races"][0]["race_id"] == 146508
    assert payload["races"][0]["offered_distances"] == ["Half", "5K"]
    assert payload["max_prediction_date"] == "2026-08-24"
    assert payload["prediction_horizon_message"] == (
        "Predictions can only be made up until 2026-08-24."
    )


def test_prediction_horizon(client):
    response = client.get("/api/prediction-horizon")
    assert response.status_code == 200
    payload = response.json()
    assert payload["max_prediction_date"] == "2026-08-24"
    assert "Predictions can only be made up until" in payload["prediction_horizon_message"]


@patch("api.routes.races.enrich_race")
def test_get_race_detail(mock_enrich, client):
    mock_enrich.return_value = EnrichedRace(
        race_id=146508,
        name="Bridge to Brew",
        city="Port Huron",
        state="MI",
        race_date=date(2026, 8, 9),
        elev_gain_ft=150,
        elev_loss_ft=150,
        elev_source="override",
        temp_f=74.0,
        weather_source="typical",
        offered_events=[
            EnrichedRaceEvent(1140659, "5K", "5K", 3.10686),
        ],
        warnings=[],
    )

    response = client.get("/api/races/146508")
    assert response.status_code == 200
    payload = response.json()
    assert payload["race_id"] == 146508
    assert payload["elev_source"] == "override"
    assert payload["offered_events"][0]["distance_label"] == "5K"


@patch("api.routes.predict.enrich_race")
@patch("api.routes.predict.predict_all")
@patch("api.routes.predict.load_model")
def test_predict_by_race_id(mock_load_model, mock_predict_all, mock_enrich, client):
    mock_load_model.return_value = type("Model", (), {"default_temp_f": 60.0})()
    mock_enrich.return_value = EnrichedRace(
        race_id=146508,
        name="Bridge to Brew",
        city="Port Huron",
        state="MI",
        race_date=date(2026, 8, 9),
        elev_gain_ft=150,
        elev_loss_ft=150,
        elev_source="override",
        temp_f=74.0,
        weather_source="typical",
        offered_events=[],
        warnings=[],
    )
    mock_predict_all.return_value = [
        RacePrediction(
            distance_label="5K",
            distance_mi=3.1,
            baseline_time_sec=1800,
            residual_sec=0,
            predicted_time_sec=1800,
            vdot_time_sec=1800,
            riegel_time_sec=1800,
            pace_min_per_mi=9.6,
            interval_low_sec=1750,
            interval_high_sec=1850,
            confidence=82,
        )
    ]

    response = client.post("/api/predict", json={"race_id": 146508})
    assert response.status_code == 200
    payload = response.json()
    assert payload["race_id"] == 146508
    assert payload["race_name"] == "Bridge to Brew"
    assert payload["elev_gain_ft"] == 150
    assert payload["temp_f"] == 74.0
    assert payload["temp_source"] == "typical"
    assert payload["predictions"][0]["confidence"] == 82


def test_predict_requires_model_file(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "activities.csv").write_text(
        "Activity Type,Activity Date,Distance,Moving Time,Activity Name\n"
        "Run,\"Jun 1, 2026\",5000,1200,Morning Run\n",
        encoding="utf-8",
    )

    def override_settings() -> Settings:
        return Settings(
            data_dir=data_dir,
            model_path=tmp_path / "missing-model.pkl",
            cors_origins=[],
            overrides_path=tmp_path / "overrides.json",
            enrichment_cache_dir=tmp_path / "cache",
            gpx_dir=tmp_path / "gpx",
        )

    app.dependency_overrides[get_settings] = override_settings
    try:
        response = TestClient(app).post(
            "/api/predict",
            json={"elev_gain_ft": 0, "elev_loss_ft": 0},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "train --population" in response.json()["detail"]


def test_predict_missing_csv(client, tmp_path):
    data_dir = tmp_path / "empty"
    data_dir.mkdir()

    def override_settings() -> Settings:
        return _settings(tmp_path, data_dir=data_dir)

    app.dependency_overrides[get_settings] = override_settings
    try:
        response = TestClient(app).post(
            "/api/predict",
            json={"elev_gain_ft": 0, "elev_loss_ft": 0},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "activities.csv" in response.json()["detail"]


@patch("api.routes.predict.load_model")
def test_predict_rejects_past_as_of(mock_load_model, client):
    mock_load_model.return_value = type("Model", (), {"default_temp_f": 60.0})()
    response = client.post(
        "/api/predict",
        json={
            "elev_gain_ft": 0,
            "elev_loss_ft": 0,
            "as_of": "2020-01-01",
        },
    )
    assert response.status_code == 400
    assert "as_of must be today" in response.json()["detail"]


@patch("api.routes.predict.load_model")
def test_predict_rejects_beyond_horizon(mock_load_model, client):
    mock_load_model.return_value = type("Model", (), {"default_temp_f": 60.0})()
    response = client.post(
        "/api/predict",
        json={
            "elev_gain_ft": 0,
            "elev_loss_ft": 0,
            "as_of": "2026-12-01",
        },
    )
    assert response.status_code == 400
    assert "Predictions can only be made up until 2026-08-24." in response.json()["detail"]


def test_predict_rejects_negative_elevation(client):
    response = client.post(
        "/api/predict",
        json={"elev_gain_ft": -1, "elev_loss_ft": 0},
    )
    assert response.status_code == 422


def test_predict_requires_manual_elevation_without_race_id(client):
    response = client.post("/api/predict", json={"elev_gain_ft": 100})
    assert response.status_code == 422
