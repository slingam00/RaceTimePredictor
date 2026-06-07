"""Multi-athlete public benchmark with per-variant MAPE comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from race_predictor.constants import RACE_DISTANCES_MI
from race_predictor.data.models import Run, TrainedModel
from race_predictor.evaluate.metrics import DistanceMetrics, compute_metrics
from race_predictor.evaluate.split import split_by_athlete
from race_predictor.features.fitness import compute_fitness_features
from race_predictor.models.baseline import predict_baseline, predict_variant_time_sec
from race_predictor.models.residual import (
    distance_label_for_mi,
    predict_residual,
    train_residual_model,
)
from race_predictor.models.ultrassignup import (
    predict_ultrassignup_time_sec,
    reference_winning_times,
)

BENCHMARK_VARIANTS = ("riegel", "vdot", "ultrassignup", "hybrid")
MIN_PRIOR_RUNS = 3


@dataclass
class VariantDistanceMetrics:
    variant: str
    distance_label: str
    count: int
    mape: float
    rmse_sec: float
    within_5_pct: float


@dataclass
class BenchmarkResult:
    corpus_runs: int
    train_athletes: int
    test_athletes: int
    train_athlete_ids: list[str]
    test_athlete_ids: list[str]
    holdout_frac: float
    seed: int
    holdout_count: int
    variants: list[str]
    metrics_by_variant: dict[str, list[DistanceMetrics]]
    comparison_table: list[VariantDistanceMetrics]
    generalization: dict[str, object] = field(default_factory=dict)


def _is_benchmark_holdout(run: Run) -> bool:
    if run.is_likely_race:
        return True
    return distance_label_for_mi(run.distance_mi) is not None


def _predict_variant(
    variant: str,
    prior_runs: list[Run],
    holdout: Run,
    distance_label: str,
    temp_f: float,
    model: TrainedModel,
    reference_times: dict[str, float],
) -> float | None:
    if variant == "riegel":
        return predict_variant_time_sec(
            prior_runs,
            holdout.date,
            distance_label,
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
            variant="riegel",
        )
    if variant == "vdot":
        return predict_variant_time_sec(
            prior_runs,
            holdout.date,
            distance_label,
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
            variant="vdot",
        )
    if variant == "ultrassignup":
        return predict_ultrassignup_time_sec(
            prior_runs,
            holdout.date,
            distance_label,
            holdout.elev_gain_ft,
            holdout.elev_loss_ft,
            temp_f,
            reference_times,
        )

    baseline = predict_baseline(
        prior_runs,
        holdout.date,
        distance_label,
        holdout.elev_gain_ft,
        holdout.elev_loss_ft,
        temp_f,
    )
    if baseline is None:
        return None

    features = compute_fitness_features(prior_runs, holdout.date, distance_label)
    residual = predict_residual(
        model,
        features,
        baseline.distance_mi,
    )
    return max(0.0, baseline.predicted_time_sec + residual)


def _evaluate_holdouts(
    runs: list[Run],
    model: TrainedModel,
    reference_times: dict[str, float],
) -> tuple[dict[str, dict[str, list[float]]], int]:
    actuals_by_variant: dict[str, dict[str, list[float]]] = {
        variant: {label: [] for label in RACE_DISTANCES_MI} for variant in BENCHMARK_VARIANTS
    }
    preds_by_variant: dict[str, dict[str, list[float]]] = {
        variant: {label: [] for label in RACE_DISTANCES_MI} for variant in BENCHMARK_VARIANTS
    }
    holdout_count = 0

    for holdout in runs:
        distance_label = distance_label_for_mi(holdout.distance_mi)
        if distance_label is None or not _is_benchmark_holdout(holdout):
            continue

        prior = [run for run in runs if run.athlete_id == holdout.athlete_id and run.date < holdout.date]
        if len(prior) < MIN_PRIOR_RUNS:
            continue

        temp_f = holdout.temp_f if holdout.temp_f is not None else model.default_temp_f
        holdout_count += 1

        for variant in BENCHMARK_VARIANTS:
            predicted = _predict_variant(
                variant,
                prior,
                holdout,
                distance_label,
                temp_f,
                model,
                reference_times,
            )
            if predicted is None or predicted <= 0:
                continue
            actuals_by_variant[variant][distance_label].append(holdout.moving_time_sec)
            preds_by_variant[variant][distance_label].append(predicted)

    return (
        {
            variant: {
                label: (actuals_by_variant[variant][label], preds_by_variant[variant][label])
                for label in RACE_DISTANCES_MI
            }
            for variant in BENCHMARK_VARIANTS
        },
        holdout_count,
    )


def _metrics_from_buckets(
    buckets: dict[str, tuple[list[float], list[float]]],
) -> list[DistanceMetrics]:
    metrics: list[DistanceMetrics] = []
    for label in RACE_DISTANCES_MI:
        actuals, preds = buckets[label]
        metrics.append(compute_metrics(label, actuals, preds))
    return metrics


def _comparison_table(metrics_by_variant: dict[str, list[DistanceMetrics]]) -> list[VariantDistanceMetrics]:
    rows: list[VariantDistanceMetrics] = []
    for variant in BENCHMARK_VARIANTS:
        for metric in metrics_by_variant[variant]:
            rows.append(
                VariantDistanceMetrics(
                    variant=variant,
                    distance_label=metric.label,
                    count=metric.count,
                    mape=metric.mape,
                    rmse_sec=metric.rmse_sec,
                    within_5_pct=metric.within_5_pct,
                )
            )
    return rows


def _weighted_mape(metrics: list[DistanceMetrics]) -> float | None:
    total = sum(metric.count for metric in metrics)
    if total == 0:
        return None
    return sum(metric.mape * metric.count for metric in metrics) / total


def _assess_generalization(
    train_metrics: dict[str, list[DistanceMetrics]],
    test_metrics: dict[str, list[DistanceMetrics]],
) -> dict[str, object]:
    hybrid_train = _weighted_mape(train_metrics["hybrid"])
    hybrid_test = _weighted_mape(test_metrics["hybrid"])
    riegel_test = _weighted_mape(test_metrics["riegel"])
    vdot_test = _weighted_mape(test_metrics["vdot"])
    ultra_test = _weighted_mape(test_metrics["ultrassignup"])

    verdict = "inconclusive"
    notes: list[str] = []

    if hybrid_test is not None and riegel_test is not None:
        if hybrid_test < riegel_test * 0.95:
            notes.append("Hybrid beats Riegel on held-out athletes.")
        else:
            notes.append("Hybrid does not clearly beat Riegel on held-out athletes.")

    if hybrid_test is not None and vdot_test is not None:
        if hybrid_test < vdot_test * 0.95:
            notes.append("Hybrid beats VDOT on held-out athletes.")
        else:
            notes.append("Hybrid does not clearly beat VDOT on held-out athletes.")

    if hybrid_test is not None and ultra_test is not None:
        if hybrid_test < ultra_test * 0.95:
            notes.append("Hybrid beats UltraSignup on held-out athletes.")
        else:
            notes.append("Hybrid does not clearly beat UltraSignup on held-out athletes.")

    if hybrid_train is not None and hybrid_test is not None:
        gap = hybrid_test - hybrid_train
        notes.append(f"Train/test hybrid MAPE gap: {gap:+.1%}.")
        if gap <= 0.02:
            verdict = "generalizes"
            notes.append("ML layer appears to generalize across athletes (small train/test gap).")
        elif gap <= 0.05:
            verdict = "mixed"
            notes.append("ML layer shows moderate athlete-specific overfitting.")
        else:
            verdict = "athlete_specific"
            notes.append("ML layer appears athlete-specific (large train/test gap).")
    else:
        notes.append("Insufficient held-out races to assess generalization.")

    best_variant = None
    best_mape = None
    for variant in BENCHMARK_VARIANTS:
        mape = _weighted_mape(test_metrics[variant])
        if mape is None:
            continue
        if best_mape is None or mape < best_mape:
            best_mape = mape
            best_variant = variant

    return {
        "verdict": verdict,
        "notes": notes,
        "hybrid_train_mape": hybrid_train,
        "hybrid_test_mape": hybrid_test,
        "riegel_test_mape": riegel_test,
        "vdot_test_mape": vdot_test,
        "ultrassignup_test_mape": ultra_test,
        "best_test_variant": best_variant,
        "best_test_mape": best_mape,
    }


def run_benchmark(
    runs: list[Run],
    holdout_frac: float = 0.25,
    seed: int = 42,
) -> BenchmarkResult:
    """Train on a subset of athletes and score all variants on held-out athletes."""
    train_runs, test_runs, train_ids, test_ids = split_by_athlete(
        runs, holdout_frac=holdout_frac, seed=seed
    )
    model = train_residual_model(train_runs)
    reference_times = reference_winning_times(train_runs)

    train_buckets, train_holdouts = _evaluate_holdouts(train_runs, model, reference_times)
    test_buckets, test_holdouts = _evaluate_holdouts(test_runs, model, reference_times)

    train_metrics = {
        variant: _metrics_from_buckets(train_buckets[variant]) for variant in BENCHMARK_VARIANTS
    }
    test_metrics = {
        variant: _metrics_from_buckets(test_buckets[variant]) for variant in BENCHMARK_VARIANTS
    }
    generalization = _assess_generalization(train_metrics, test_metrics)

    return BenchmarkResult(
        corpus_runs=len(runs),
        train_athletes=len(train_ids),
        test_athletes=len(test_ids),
        train_athlete_ids=train_ids,
        test_athlete_ids=test_ids,
        holdout_frac=holdout_frac,
        seed=seed,
        holdout_count=test_holdouts,
        variants=list(BENCHMARK_VARIANTS),
        metrics_by_variant=test_metrics,
        comparison_table=_comparison_table(test_metrics),
        generalization=generalization,
    )


def benchmark_to_dict(result: BenchmarkResult) -> dict:
    return {
        "corpus_runs": result.corpus_runs,
        "train_athletes": result.train_athletes,
        "test_athletes": result.test_athletes,
        "train_athlete_ids": result.train_athlete_ids,
        "test_athlete_ids": result.test_athlete_ids,
        "holdout_frac": result.holdout_frac,
        "seed": result.seed,
        "holdout_count": result.holdout_count,
        "variants": result.variants,
        "metrics_by_variant": {
            variant: [asdict(metric) for metric in metrics]
            for variant, metrics in result.metrics_by_variant.items()
        },
        "comparison_table": [asdict(row) for row in result.comparison_table],
        "generalization": result.generalization,
    }
