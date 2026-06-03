"""Time-series backtest for hybrid predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from race_predictor.constants import RACE_DISTANCES_MI
from race_predictor.confidence.scoring import prediction_interval
from race_predictor.data.models import Run, TrainedModel
from race_predictor.evaluate.metrics import DistanceMetrics, compute_metrics
from race_predictor.features.fitness import compute_fitness_features
from race_predictor.models.baseline import predict_baseline
from race_predictor.models.residual import (
    distance_label_for_mi,
    predict_residual,
    train_residual_model,
)


@dataclass
class BacktestResult:
    holdout_count: int
    training_rows: int
    metrics: list[DistanceMetrics]
    holdouts: list[dict]


def run_backtest(runs: list[Run], model: TrainedModel | None = None) -> BacktestResult:
    if model is None:
        model = train_residual_model(runs)

    bucket_actuals: dict[str, list[float]] = {label: [] for label in RACE_DISTANCES_MI}
    bucket_preds: dict[str, list[float]] = {label: [] for label in RACE_DISTANCES_MI}
    bucket_interval_hits: dict[str, list[bool]] = {label: [] for label in RACE_DISTANCES_MI}
    holdouts: list[dict] = []

    for holdout in runs:
        label = distance_label_for_mi(holdout.distance_mi)
        if label is None and not holdout.is_likely_race:
            continue
        if label is None:
            continue

        prior = [run for run in runs if run.date < holdout.date]
        if len(prior) < 3:
            continue

        temp_f = holdout.temp_f if holdout.temp_f is not None else model.default_temp_f
        baseline = predict_baseline(
            prior,
            holdout.date,
            label,
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
        )
        if baseline is None:
            continue

        features = compute_fitness_features(prior, holdout.date, label)
        residual = predict_residual(
            model,
            features,
            baseline.distance_mi,
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
        )
        predicted = max(0.0, baseline.predicted_time_sec + residual)
        low, high = prediction_interval(model, label, predicted, features)
        in_interval = low <= holdout.moving_time_sec <= high

        bucket_actuals[label].append(holdout.moving_time_sec)
        bucket_preds[label].append(predicted)
        bucket_interval_hits[label].append(in_interval)

        holdouts.append(
            {
                "date": holdout.date.isoformat(),
                "name": holdout.name,
                "distance_label": label,
                "distance_mi": holdout.distance_mi,
                "actual_sec": holdout.moving_time_sec,
                "predicted_sec": predicted,
                "baseline_sec": baseline.predicted_time_sec,
                "interval_low_sec": low,
                "interval_high_sec": high,
                "in_80_interval": in_interval,
            }
        )

    metrics: list[DistanceMetrics] = []
    for label in RACE_DISTANCES_MI:
        result = compute_metrics(label, bucket_actuals[label], bucket_preds[label])
        hits = bucket_interval_hits[label]
        if hits:
            result.interval_coverage_80 = sum(hits) / len(hits)
        metrics.append(result)

    from race_predictor.models.residual import build_training_rows

    _x, _y, meta = build_training_rows(runs)

    return BacktestResult(
        holdout_count=len(holdouts),
        training_rows=len(meta),
        metrics=metrics,
        holdouts=holdouts,
    )


def backtest_to_dict(result: BacktestResult) -> dict:
    return {
        "holdout_count": result.holdout_count,
        "training_rows": result.training_rows,
        "metrics": [asdict(m) for m in result.metrics],
        "holdouts": result.holdouts,
    }
