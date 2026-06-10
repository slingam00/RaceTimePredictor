"""Command-line interface for race-predictor."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import click

from race_predictor.data.benchmark_loader import load_benchmark_corpus
from race_predictor.data.loader import load_runs
from race_predictor.data.race_enrichment import (
    DEFAULT_CACHE_DIR,
    DEFAULT_GPX_DIR,
    DEFAULT_OVERRIDES_PATH,
    enrich_race,
)
from race_predictor.data.runsignup_client import (
    RunSignupClient,
    RunSignupError,
    has_runsignup_auth,
)
from race_predictor.data.runsignup_login import RunSignupLoginError, login_and_save
from race_predictor.data.runsignup_oauth import (
    DEFAULT_TOKEN_PATH,
    access_token_from_env_or_file,
    oauth_config_from_env,
)
from race_predictor.data.runsignup_corpus import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_COVERAGE_REPORT_PATH,
    format_coverage_summary,
    sync_corpus,
)
from race_predictor.evaluate.backtest import backtest_to_dict, run_backtest
from race_predictor.evaluate.benchmark import benchmark_to_dict, run_benchmark
from race_predictor.formatting import format_pace, format_time
from race_predictor.features.prediction_horizon import (
    format_prediction_horizon_message,
    max_prediction_date,
)
from race_predictor.models.predictor import (
    DEFAULT_MODEL_PATH,
    load_model,
    predict_all,
    predict_race,
    train,
    train_population,
)

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


def _format_interval(low_sec: float, high_sec: float) -> str:
    return f"{format_time(low_sec)} – {format_time(high_sec)}"


def _parse_as_of_date(as_of: str) -> date:
    if not _AS_OF_DATE_RE.match(as_of):
        raise click.ClickException(
            f"The date {as_of!r} is invalid. Use YYYY-MM-DD (e.g. 2026-06-15)."
        )
    try:
        return datetime.strptime(as_of, "%Y-%m-%d").date()
    except ValueError as exc:
        raise click.ClickException(
            f"The date {as_of!r} is invalid. Use YYYY-MM-DD (e.g. 2026-06-15)."
        ) from exc


def _emit_predictions(
    *,
    race_name: str | None,
    as_of_dt: datetime,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float,
    predictions: list,
    warnings: list[str] | None = None,
) -> None:
    if race_name:
        click.echo(f"\n{race_name}")
    if warnings:
        for warning in warnings:
            click.echo(f"Warning: {warning}")
    click.echo(
        f"\nPredictions as of {as_of_dt.date()} "
        f"(+{elev_gain_ft:.0f}/-{elev_loss_ft:.0f} ft, {temp_f:.1f}°F)\n"
    )
    click.echo(
        f"{'Distance':<10} {'Predicted':<12} {'Pace':<12} "
        f"{'80% Interval':<22} {'Conf'}"
    )
    click.echo("-" * 68)
    for prediction in predictions:
        interval = _format_interval(
            prediction.interval_low_sec,
            prediction.interval_high_sec,
        )
        click.echo(
            f"{prediction.distance_label:<10} "
            f"{format_time(prediction.predicted_time_sec):<12} "
            f"{format_pace(prediction.pace_min_per_mi):<12} "
            f"{interval:<22} "
            f"{prediction.confidence}"
        )


@click.group()
def main() -> None:
    """Predict race times from Strava training data (US customary units)."""


@main.command("runsignup-login")
@click.option(
    "--token-path",
    default=str(DEFAULT_TOKEN_PATH),
    show_default=True,
    type=click.Path(),
    help="Where to store the OAuth access token.",
)
def runsignup_login_cmd(token_path: str) -> None:
    """Log in to RunSignup via browser OAuth and save an API access token."""
    config = oauth_config_from_env()
    if config is None:
        raise click.ClickException(
            "Set RUNSIGNUP_CLIENT_ID and RUNSIGNUP_CLIENT_SECRET in .env "
            "(from https://runsignup.com/Profile/OAuth2 → App Development)."
        )
    click.echo(f"Using redirect URI: {config.redirect_uri}")
    click.echo(
        "Ensure this exact URI is registered on your RunSignup OAuth client."
    )
    try:
        tokens = login_and_save(config, token_path)
    except RunSignupLoginError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"RunSignup login failed: {exc}") from exc

    expiry = tokens.expires_at.isoformat() if tokens.expires_at else "unknown"
    click.echo(f"Login successful. Token saved to {token_path}")
    click.echo(f"Access token expires: {expiry}")
    click.echo("Next: race-predictor sync-corpus")


@main.command("sync-corpus")
@click.option(
    "--catalog",
    default=str(DEFAULT_CATALOG_PATH),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Curated RunSignup race catalog (catalog/races.json).",
)
@click.option(
    "--output",
    default=str(DEFAULT_CORPUS_PATH),
    show_default=True,
    help="Output CSV path for the multi-athlete training corpus.",
)
@click.option(
    "--coverage-report",
    default=str(DEFAULT_COVERAGE_REPORT_PATH),
    show_default=True,
    help="JSON report with finisher counts per distance and course type.",
)
@click.option(
    "--max-finishers",
    default=1000,
    show_default=True,
    type=int,
    help="Maximum finishers to fetch per event.",
)
@click.option(
    "--require-coverage",
    is_flag=True,
    default=False,
    help="Exit with error if distance/course coverage thresholds are not met.",
)
def sync_corpus_cmd(
    catalog: str,
    output: str,
    coverage_report: str,
    max_finishers: int,
    require_coverage: bool,
) -> None:
    """Collect finisher results from curated RunSignup races into a training corpus."""
    client = RunSignupClient()
    try:
        runs, report = sync_corpus(
            client,
            catalog_path=catalog,
            output_path=output,
            coverage_report_path=coverage_report,
            max_finishers_per_event=max_finishers,
        )
    except RunSignupError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(format_coverage_summary(report))
    click.echo(f"\nCorpus saved to {output}")
    click.echo(f"Coverage report saved to {coverage_report}")

    if report.race_stats:
        click.echo("\nPer-race sync:")
        for stats in report.race_stats:
            if stats.get("error"):
                click.echo(f"  FAIL {stats['catalog_entry']}: {stats['error']}")
            else:
                click.echo(
                    f"  OK   {stats['catalog_entry']}: "
                    f"{stats.get('finishers', 0)} finishers"
                )

    if require_coverage and not report.passed:
        raise click.ClickException(
            "Coverage thresholds not met. See coverage report for gaps."
        )

    if not runs:
        raise click.ClickException("No finisher results collected.")


@main.command("train")
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option(
    "--corpus",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="RunSignup finisher corpus CSV for population training.",
)
@click.option(
    "--population/--athlete",
    "population_mode",
    default=None,
    help="Train on population corpus (default) or a single-athlete Strava export.",
)
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
def train_cmd(
    data_dir: str,
    corpus: str | None,
    population_mode: bool | None,
    model_path: str,
) -> None:
    """Train the hybrid model from a population corpus or Strava export."""
    if population_mode is False:
        use_population = False
    elif population_mode is True or corpus is not None:
        use_population = True
    else:
        use_population = Path(DEFAULT_CORPUS_PATH).is_file()

    if use_population:
        corpus_path = Path(corpus or DEFAULT_CORPUS_PATH)
        if not corpus_path.is_file():
            raise click.ClickException(
                f"Population corpus not found at {corpus_path}. "
                "Run `race-predictor sync-corpus` first or pass --corpus."
            )
        runs = load_benchmark_corpus(corpus_path)
        if not runs:
            raise click.ClickException(f"No runs found in population corpus {corpus_path}")
        model = train_population(runs, model_path)
        mode_label = "population"
    else:
        csv_path = _require_activities_csv(data_dir)
        runs = load_runs(csv_path)
        if not runs:
            raise click.ClickException(f"No runs found in {csv_path}")
        model = train(runs, model_path)
        mode_label = "athlete"

    buckets = ", ".join(sorted(model.residual_stats or {})) or "none"
    click.echo(f"Training mode: {mode_label}")
    click.echo(f"Trained on {len(runs)} runs.")
    click.echo(f"Model saved to {model_path}")
    click.echo(f"Default temperature: {model.default_temp_f:.1f}°F")
    click.echo(f"Residual buckets: {buckets}")


@main.command()
@click.option("--data-dir", default="data", show_default=True, type=click.Path(exists=True))
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), show_default=True)
@click.option(
    "--race-id",
    type=int,
    default=None,
    help="RunSignup race ID; fetches course elevation and weather automatically.",
)
@click.option(
    "--event-id",
    type=int,
    default=None,
    help="Predict a single event distance when used with --race-id.",
)
@click.option(
    "--elev-gain-ft",
    default=None,
    type=float,
    help="Race elevation gain in feet (required without --race-id).",
)
@click.option(
    "--elev-loss-ft",
    default=None,
    type=float,
    help="Race elevation loss in feet (required without --race-id).",
)
@click.option("--temp-f", default=None, type=float, help="Race temperature in °F.")
@click.option(
    "--as-of",
    default=None,
    help="Prediction date (YYYY-MM-DD). Must be today or later. Defaults to latest run date.",
)
def predict(
    data_dir: str,
    model_path: str,
    race_id: int | None,
    event_id: int | None,
    elev_gain_ft: float | None,
    elev_loss_ft: float | None,
    temp_f: float | None,
    as_of: str | None,
) -> None:
    """Predict race times for 5K, 10K, Half, and Marathon."""
    if race_id is None and (elev_gain_ft is None or elev_loss_ft is None):
        raise click.ClickException(
            "--elev-gain-ft and --elev-loss-ft are required when --race-id is omitted."
        )
    if event_id is not None and race_id is None:
        raise click.ClickException("--event-id requires --race-id.")

    as_of_date: date | None = _parse_as_of_date(as_of) if as_of else None
    if as_of_date is not None and as_of_date < date.today():
        raise click.ClickException(
            f"--as-of must be today ({date.today():%Y-%m-%d}) or a future date; got {as_of}."
        )

    if elev_gain_ft is not None or elev_loss_ft is not None:
        _validate_race_conditions(
            elev_gain_ft if elev_gain_ft is not None else 0.0,
            elev_loss_ft if elev_loss_ft is not None else 0.0,
            temp_f,
        )
    elif temp_f is not None:
        _validate_race_conditions(0.0, 0.0, temp_f)

    csv_path = _require_activities_csv(data_dir)
    runs = load_runs(csv_path)
    if not runs:
        raise click.ClickException(f"No runs found in {csv_path}")

    horizon = max_prediction_date(runs)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        if race_id is not None:
            raise click.ClickException(
                "No trained model found. Run `race-predictor train --population` first."
            )
        click.echo("No trained model found; training now...")
        model = train(runs, model_path)
    else:
        model = load_model(model_path)

    race_name: str | None = None
    warnings: list[str] = []
    distance_labels: list[str] | None = None

    if race_id is not None:
        client = RunSignupClient()
        try:
            enriched = enrich_race(
                client,
                race_id,
                as_of=as_of_date,
                overrides_path=DEFAULT_OVERRIDES_PATH,
                cache_dir=DEFAULT_CACHE_DIR,
                gpx_dir=DEFAULT_GPX_DIR,
            )
        except RunSignupError as exc:
            raise click.ClickException(str(exc)) from exc

        race_name = enriched.name
        warnings = list(enriched.warnings)
        if elev_gain_ft is None:
            elev_gain_ft = enriched.elev_gain_ft
        if elev_loss_ft is None:
            elev_loss_ft = enriched.elev_loss_ft
        if temp_f is None and enriched.temp_f is not None:
            temp_f = enriched.temp_f
        if as_of_date is None:
            as_of_date = enriched.race_date

        if event_id is not None:
            matching = [
                event
                for event in enriched.offered_events
                if event.event_id == event_id
            ]
            if not matching:
                raise click.ClickException(
                    f"event_id {event_id} not found for race {race_id}."
                )
            distance_labels = [matching[0].distance_label]

    if elev_gain_ft is None or elev_loss_ft is None:
        raise click.ClickException(
            "Elevation is unknown for this race. Provide --elev-gain-ft and --elev-loss-ft."
        )
    _validate_race_conditions(elev_gain_ft, elev_loss_ft, temp_f)

    if as_of_date is None:
        as_of_date = runs[-1].date.date()
    if horizon is not None and as_of_date > horizon:
        raise click.ClickException(format_prediction_horizon_message(horizon))

    as_of_dt = datetime.combine(as_of_date, datetime.min.time())
    if temp_f is None:
        temp_f = model.default_temp_f

    if distance_labels:
        predictions = []
        for label in distance_labels:
            prediction = predict_race(
                runs,
                model,
                as_of_dt,
                label,
                elev_gain_ft=elev_gain_ft,
                elev_loss_ft=elev_loss_ft,
                temp_f=temp_f,
            )
            if prediction is not None:
                predictions.append(prediction)
    else:
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

    if horizon is not None:
        click.echo(format_prediction_horizon_message(horizon))

    _emit_predictions(
        race_name=race_name,
        as_of_dt=as_of_dt,
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        temp_f=temp_f,
        predictions=predictions,
        warnings=warnings or None,
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
