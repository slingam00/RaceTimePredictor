"""Evaluation metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DistanceMetrics:
    label: str
    count: int
    mape: float
    rmse_sec: float
    median_abs_error_sec: float
    within_3_pct: float
    within_5_pct: float
    within_10_pct: float
    interval_coverage_80: float | None


def compute_metrics(label: str, actuals: list[float], predicted: list[float]) -> DistanceMetrics:
    if not actuals:
        return DistanceMetrics(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)

    actual = np.array(actuals)
    pred = np.array(predicted)
    pct_errors = np.abs(actual - pred) / actual
    abs_errors = np.abs(actual - pred)

    return DistanceMetrics(
        label=label,
        count=len(actuals),
        mape=float(np.mean(pct_errors)),
        rmse_sec=float(np.sqrt(np.mean(abs_errors**2))),
        median_abs_error_sec=float(np.median(abs_errors)),
        within_3_pct=float(np.mean(pct_errors <= 0.03)),
        within_5_pct=float(np.mean(pct_errors <= 0.05)),
        within_10_pct=float(np.mean(pct_errors <= 0.10)),
        interval_coverage_80=None,
    )
