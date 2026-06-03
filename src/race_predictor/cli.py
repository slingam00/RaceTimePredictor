"""Command-line interface for race-predictor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from race_predictor.data.loader import load_runs
from race_predictor.evaluate.backtest import backtest_to_dict, run_backtest
from race_predictor.formatting import format_pace, format_time
from race_predictor.models.predictor import DEFAULT_MODEL_PATH, load_model, predict_all, train


@click.group()
def main() -> None:
    """Predict race times from Strava training data (US customary units)."""


@main.command("train")
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
def train_cmd(data_dir: str, model_path: str) -> None:
    """Train the hybrid model from a Strava export."""
    csv_path = Path(data_dir) / "activities.csv"
    runs = load_runs(csv_path)
    if not runs:
        raise click.ClickException(f"No runs found in {csv_path}")

    model = train(runs, model_path)
    buckets = ", ".join(sorted(model.residual_stats or {})) or "none"
    click.echo(f"Trained on {len(runs)} runs.")
    click.echo(f"Model saved to {model_path}")
    click.echo(f"Default temperature: {model.default_temp_f:.1f}°F")
    click.echo(f"Backtest buckets: {buckets}")


@main.command()
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
@click.option("--elev-gain-ft", required=True, type=float, help="Race elevation gain in feet.")
@click.option("--elev-loss-ft", required=True, type=float, help="Race elevation loss in feet.")
@click.option("--temp-f", default=None, type=float, help="Race temperature in °F.")
@click.option(
    "--as-of",
    default=None,
    help="Prediction date (YYYY-MM-DD). Defaults to latest run date.",
)
def predict(
    data_dir: str,
    model_path: str,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None,
    as_of: str | None,
) -> None:
    """Predict race times for 5K, 10K, Half, and Marathon."""
    csv_path = Path(data_dir) / "activities.csv"
    runs = load_runs(csv_path)
    if not runs:
        raise click.ClickException(f"No runs found in {csv_path}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        click.echo("No trained model found; training now...")
        model = train(runs, model_path)
    else:
        model = load_model(model_path)

    if as_of:
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    else:
        as_of_dt = runs[-1].date

    predictions = predict_all(
        runs,
        model,
        as_of_dt,
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        temp_f=temp_f,
    )

    click.echo(f"\nPredictions as of {as_of_dt.date()} (US customary units)\n")
    click.echo(
        f"{'Distance':<10} {'Predicted':<12} {'Pace':<12} "
        f"{'80% Interval':<22} {'Conf'}"
    )
    click.echo("-" * 68)
    for p in predictions:
        interval = f"{format_time(p.interval_low_sec)} – {format_time(p.interval_high_sec)}"
        click.echo(
            f"{p.distance_label:<10} "
            f"{format_time(p.predicted_time_sec):<12} "
            f"{format_pace(p.pace_min_per_mi):<12} "
            f"{interval:<22} "
            f"{p.confidence}"
        )


@main.command("evaluate")
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
@click.option("--output", default="reports/backtest.json", show_default=True)
def evaluate_cmd(data_dir: str, model_path: str, output: str) -> None:
    """Run time-series backtest and print metrics."""
    csv_path = Path(data_dir) / "activities.csv"
    runs = load_runs(csv_path)
    if not runs:
        raise click.ClickException(f"No runs found in {csv_path}")

    model = load_model(model_path) if Path(model_path).exists() else train(runs, model_path)
    result = run_backtest(runs, model)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(backtest_to_dict(result), handle, indent=2)

    click.echo(f"Backtest holdouts: {result.holdout_count}")
    click.echo(f"Training rows: {result.training_rows}\n")
    click.echo(f"{'Distance':<10} {'N':<5} {'MAPE':<8} {'RMSE':<10} {'±5%':<8} {'80% Cov'}")
    click.echo("-" * 55)
    for m in result.metrics:
        cov = f"{m.interval_coverage_80:.0%}" if m.interval_coverage_80 is not None else "—"
        click.echo(
            f"{m.label:<10} {m.count:<5} {m.mape:.1%}   {m.rmse_sec:>6.0f}s   "
            f"{m.within_5_pct:.0%}     {cov}"
        )
    click.echo(f"\nReport saved to {out_path}")
