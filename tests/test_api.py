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


@pytest.fixture
def client(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "activities.csv").write_text(
        "Activity Type,Activity Date,Distance,Moving Time,Activity Name\n"
        "Run,\"Jun 1, 2026\",5000,1200,Morning Run\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pkl"

    def override_settings() -> Settings:
        return Settings(
            data_dir=data_dir,
            model_path=model_path,
            cors_origins=["http://localhost:3000"],
        )

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


def test_predict_missing_csv(client, tmp_path):
    data_dir = tmp_path / "empty"
    data_dir.mkdir()

    def override_settings() -> Settings:
        return Settings(
            data_dir=data_dir,
            model_path=tmp_path / "model.pkl",
            cors_origins=[],
        )

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


def test_predict_rejects_past_as_of(client):
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


def test_predict_rejects_negative_elevation(client):
    response = client.post(
        "/api/predict",
        json={"elev_gain_ft": -1, "elev_loss_ft": 0},
    )
    assert response.status_code == 422
