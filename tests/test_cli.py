"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from race_predictor.cli import main

DATA_DIR = Path("data")


@pytest.fixture
def runner():
    return CliRunner()


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
