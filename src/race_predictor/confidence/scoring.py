"""Confidence scores and prediction intervals."""

from __future__ import annotations

from race_predictor.constants import RACE_DISTANCES_MI
from race_predictor.data.models import TrainedModel


def extrapolation_penalty(target_distance_mi: float, longest_run_mi: float) -> float:
    if longest_run_mi <= 0:
        return 40.0
    ratio = target_distance_mi / longest_run_mi
    if ratio <= 1.0:
        return 0.0
    if ratio <= 1.5:
        return 5.0 * (ratio - 1.0)
    if ratio <= 2.0:
        return 7.5 + 10.0 * (ratio - 1.5)
    return min(35.0, 17.5 + 8.0 * (ratio - 2.0))


def data_sufficiency_score(
    run_count: float,
    total_miles: float,
    days_since_last_run: float,
) -> float:
    run_score = min(30.0, run_count * 1.5)
    mileage_score = min(30.0, total_miles / 2.0)
    recency_score = max(0.0, 20.0 - days_since_last_run)
    return run_score + mileage_score + recency_score


def model_agreement_score(
    vdot_time: float | None,
    riegel_time: float | None,
    final_time: float,
) -> float:
    times = [t for t in (vdot_time, riegel_time, final_time) if t is not None and t > 0]
    if len(times) < 2:
        return 10.0
    spread = (max(times) - min(times)) / min(times)
    return max(0.0, 20.0 - spread * 100.0)


def backtest_reliability_score(model: TrainedModel, distance_label: str) -> float:
    stats = model.residual_stats.get(distance_label)
    if not stats or stats.get("count", 0) < 2:
        if model.residual_stats:
            stats = next(iter(model.residual_stats.values()))
        else:
            return 5.0
    mape = stats.get("mape", 0.15)
    return max(0.0, 20.0 - mape * 100.0)


def confidence_score(
    model: TrainedModel,
    distance_label: str,
    features: dict[str, float],
    vdot_time: float | None,
    riegel_time: float | None,
    final_time: float,
) -> int:
    target_mi = RACE_DISTANCES_MI[distance_label]
    data = data_sufficiency_score(
        features["run_count"],
        features["total_miles"],
        features["days_since_last_run"],
    )
    extrap = extrapolation_penalty(target_mi, features["longest_run_mi"])
    agreement = model_agreement_score(vdot_time, riegel_time, final_time)
    backtest = backtest_reliability_score(model, distance_label)
    raw = data + agreement + backtest - extrap
    return int(max(0, min(100, round(raw))))


def prediction_interval(
    model: TrainedModel,
    distance_label: str,
    predicted_time_sec: float,
    features: dict[str, float],
    z: float = 1.28,
) -> tuple[float, float]:
    stats = model.residual_stats.get(distance_label)
    if stats and stats.get("count", 0) >= 2:
        mape = stats["mape"]
    elif model.residual_stats:
        mape = max(s.get("mape", 0.1) for s in model.residual_stats.values())
    else:
        mape = 0.08

    target_mi = RACE_DISTANCES_MI[distance_label]
    extrap = extrapolation_penalty(target_mi, features["longest_run_mi"]) / 100.0
    margin = predicted_time_sec * (mape + extrap) * z
    low = max(0.0, predicted_time_sec - margin)
    high = predicted_time_sec + margin
    return low, high
