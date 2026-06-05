"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from race_predictor.cli import main

DATA_DIR = Path("data")


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize(
    "as_of",
    ["not-a-date", "2026-02-30", "06/04/2026", "2026-6-4"],
)
def test_predict_rejects_invalid_as_of(runner, tmp_path, as_of: str):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "activities.csv").write_text("Activity Type\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--elev-gain-ft",
            "0",
            "--elev-loss-ft",
            "0",
            "--as-of",
            as_of,
        ],
    )
    assert result.exit_code != 0
    assert "is invalid" in result.output


def test_train_requires_activities_csv(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(main, ["train", "--data-dir", str(data_dir)])
    assert result.exit_code != 0
    assert "No activities.csv found" in result.output


def test_predict_requires_activities_csv(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--elev-gain-ft",
            "0",
            "--elev-loss-ft",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "No activities.csv found" in result.output


@pytest.mark.parametrize("flag,value", [("--elev-gain-ft", "-1"), ("--elev-loss-ft", "-10")])
def test_predict_rejects_negative_elevation(runner, flag, value):
    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            "data",
            "--elev-gain-ft",
            "0",
            "--elev-loss-ft",
            "0",
            flag,
            value,
        ],
    )
    assert result.exit_code != 0
    assert "must be >= 0" in result.output


@pytest.mark.parametrize("temp_f", ["-21", "121"])
def test_predict_rejects_out_of_range_temp(runner, temp_f):
    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            "data",
            "--elev-gain-ft",
            "0",
            "--elev-loss-ft",
            "0",
            "--temp-f",
            temp_f,
        ],
    )
    assert result.exit_code != 0
    assert "--temp-f must be between" in result.output


@patch("race_predictor.cli.predict_all", return_value=[])
def test_predict_rejects_empty_predictions(mock_predict_all, runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "activities.csv").write_text(
        "Activity Type,Activity Date,Distance,Moving Time,Activity Name\n"
        "Run,\"Jun 1, 2026\",5000,1200,Morning Run\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pkl"

    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--model-path",
            str(model_path),
            "--elev-gain-ft",
            "0",
            "--elev-loss-ft",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "Could not produce predictions" in result.output
    mock_predict_all.assert_called_once()


def test_predict_rejects_past_as_of(runner, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "activities.csv").write_text("Activity Type\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--elev-gain-ft",
            "0",
            "--elev-loss-ft",
            "0",
            "--as-of",
            "2000-01-01",
        ],
    )
    assert result.exit_code != 0
    assert "must be today" in result.output


@pytest.mark.skipif(not (DATA_DIR / "activities.csv").exists(), reason="Strava export not present")
def test_cli_train_predict_evaluate(runner, tmp_path):
    model_path = tmp_path / "model.pkl"
    report_path = tmp_path / "backtest.json"

    result = runner.invoke(main, ["train", "--data-dir", str(DATA_DIR), "--model-path", str(model_path)])
    assert result.exit_code == 0, result.output
    assert model_path.exists()

    result = runner.invoke(
        main,
        [
            "predict",
            "--data-dir",
            str(DATA_DIR),
            "--model-path",
            str(model_path),
            "--elev-gain-ft",
            "492",
            "--elev-loss-ft",
            "492",
            "--temp-f",
            "72",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "5K" in result.output
    assert "Marathon" in result.output

    result = runner.invoke(
        main,
        [
            "evaluate",
            "--data-dir",
            str(DATA_DIR),
            "--model-path",
            str(model_path),
            "--output",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report_path.exists()
