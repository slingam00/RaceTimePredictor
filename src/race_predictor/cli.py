"""Command-line interface for race-predictor."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import click

from race_predictor.data.benchmark_loader import load_benchmark_corpus
from race_predictor.data.loader import load_runs
from race_predictor.evaluate.backtest import backtest_to_dict, run_backtest
from race_predictor.evaluate.benchmark import benchmark_to_dict, run_benchmark
from race_predictor.formatting import format_pace, format_time
from race_predictor.models.predictor import DEFAULT_MODEL_PATH, load_model, predict_all, train

_AS_OF_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MIN_TEMP_F = -20.0
_MAX_TEMP_F = 120.0


def _activities_csv_path(data_dir: str) -> Path:
    return Path(data_dir) / "activities.csv"


def _require_activities_csv(data_dir: str) -> Path:
    csv_path = _activities_csv_path(data_dir)
    if not csv_path.is_file():
        raise click.ClickException(
            f"No activities.csv found in {data_dir}/. Expected a Strava export file."
        )
    return csv_path


def _validate_race_conditions(
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None,
) -> None:
    if elev_gain_ft < 0:
        raise click.ClickException("--elev-gain-ft must be >= 0.")
    if elev_loss_ft < 0:
        raise click.ClickException("--elev-loss-ft must be >= 0.")
    if temp_f is not None and not (_MIN_TEMP_F <= temp_f <= _MAX_TEMP_F):
        raise click.ClickException(
            f"--temp-f must be between {_MIN_TEMP_F:.0f}°F and {_MAX_TEMP_F:.0f}°F; got {temp_f}."
        )


def _parse_as_of_date(as_of: str) -> datetime:
    if not _AS_OF_DATE_RE.match(as_of):
        raise click.ClickException(
            f"The date {as_of!r} is invalid. Use YYYY-MM-DD (e.g. 2026-06-15)."
        )
    try:
        return datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError as exc:
        raise click.ClickException(
            f"The date {as_of!r} is invalid. Use YYYY-MM-DD (e.g. 2026-06-15)."
        ) from exc


@click.group()
def main() -> None:
    """Predict race times from Strava training data (US customary units)."""


@main.command("train")
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
def train_cmd(data_dir: str, model_path: str) -> None:
    """Train the hybrid model from a Strava export."""
    csv_path = _require_activities_csv(data_dir)
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
    help="Prediction date (YYYY-MM-DD). Must be today or later. Defaults to latest run date.",
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
    as_of_dt: datetime | None = None
    if as_of:
        as_of_dt = _parse_as_of_date(as_of)
        if as_of_dt.date() < date.today():
            raise click.ClickException(
                f"--as-of must be today ({date.today():%Y-%m-%d}) or a future date; got {as_of}."
            )

    _validate_race_conditions(elev_gain_ft, elev_loss_ft, temp_f)
    csv_path = _require_activities_csv(data_dir)
    runs = load_runs(csv_path)
    if not runs:
        raise click.ClickException(f"No runs found in {csv_path}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        click.echo("No trained model found; training now...")
        model = train(runs, model_path)
    else:
        model = load_model(model_path)

    if as_of_dt is None:
        as_of_dt = runs[-1].date

    predictions = predict_all(
        runs,
        model,
        as_of_dt,
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        temp_f=temp_f,
    )
    if not predictions:
        raise click.ClickException(
            "Could not produce predictions — insufficient training data in the lookback window."
        )

    click.echo(f"\nPredictions as of {as_of_dt.date()} (US customary units)\n")
    click.echo(f"{'Distance':<10} {'Predicted Time':<14} {'Pace':<12}")
    click.echo("-" * 38)
    for p in predictions:
        click.echo(
            f"{p.distance_label:<10} "
            f"{format_time(p.predicted_time_sec):<14} "
            f"{format_pace(p.pace_min_per_mi):<12}"
        )


@main.command("evaluate")
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
@click.option("--output", default="reports/backtest.json", show_default=True)
def evaluate_cmd(data_dir: str, model_path: str, output: str) -> None:
    """Run time-series backtest and print metrics."""
    csv_path = _require_activities_csv(data_dir)
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
    click.echo(f"{'Distance':<10} {'N':<5} {'MAPE':<8} {'RMSE':<10} {'±5%':<8} {'95% Cov'}")
    click.echo("-" * 55)
    for m in result.metrics:
        cov = f"{m.interval_coverage_95:.0%}" if m.interval_coverage_95 is not None else "—"
        click.echo(
            f"{m.label:<10} {m.count:<5} {m.mape:.1%}   {m.rmse_sec:>6.0f}s   "
            f"{m.within_5_pct:.0%}     {cov}"
        )
    click.echo(f"\nReport saved to {out_path}")


@main.command("benchmark")
@click.option(
    "--corpus",
    default="benchmarks/sample_corpus.csv",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Multi-athlete benchmark CSV (see data/benchmark_loader.py columns).",
)
@click.option(
    "--holdout-frac",
    default=0.25,
    show_default=True,
    type=click.FloatRange(0.1, 0.5),
    help="Fraction of athletes held out for testing.",
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--output", default="reports/benchmark.json", show_default=True)
def benchmark_cmd(corpus: str, holdout_frac: float, seed: int, output: str) -> None:
    """Run athlete-level public benchmark across model variants."""
    try:
        runs = load_benchmark_corpus(corpus)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not runs:
        raise click.ClickException(f"No runs found in benchmark corpus {corpus}")

    try:
        result = run_benchmark(runs, holdout_frac=holdout_frac, seed=seed)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(benchmark_to_dict(result), handle, indent=2)

    click.echo(
        f"Benchmark corpus: {len(runs)} runs, "
        f"{result.train_athletes} train / {result.test_athletes} test athletes"
    )
    click.echo(f"Held-out race efforts: {result.holdout_count}\n")
    click.echo(f"{'Variant':<12} {'Distance':<10} {'N':<5} {'MAPE':<8} {'RMSE':<10} {'±5%'}")
    click.echo("-" * 60)
    for row in result.comparison_table:
        if row.count == 0:
            continue
        click.echo(
            f"{row.variant:<12} {row.distance_label:<10} {row.count:<5} "
            f"{row.mape:.1%}   {row.rmse_sec:>6.0f}s   {row.within_5_pct:.0%}"
        )

    gen = result.generalization
    click.echo("\nGeneralization")
    click.echo(f"  Verdict: {gen.get('verdict', 'inconclusive')}")
    click.echo(f"  Best test variant: {gen.get('best_test_variant')} "
               f"(MAPE {gen.get('best_test_mape', 0):.1%})")
    for note in gen.get("notes", []):
        click.echo(f"  - {note}")
    click.echo(f"\nReport saved to {out_path}")
