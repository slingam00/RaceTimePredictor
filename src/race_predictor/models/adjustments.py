"""Elevation and temperature course adjustments (US units)."""

from __future__ import annotations

from race_predictor.constants import OPTIMAL_TEMP_F


def temperature_factor(temp_f: float, optimal_temp_f: float = OPTIMAL_TEMP_F) -> float:
    """Daniels-style heat penalty: ~0.5% per °F above optimal."""
    if temp_f <= optimal_temp_f:
        return 1.0
    return 1.0 + 0.005 * (temp_f - optimal_temp_f)


def elevation_time_adjustment_sec(
    distance_mi: float,
    elev_gain_ft: float,
    elev_loss_ft: float,
    sec_per_100ft: float = 4.5,
) -> float:
    """Add time for net course elevation vs flat baseline (US units)."""
    if distance_mi <= 0:
        return 0.0
    net_gain_ft = elev_gain_ft - elev_loss_ft
    return (net_gain_ft / 100.0) * sec_per_100ft


def apply_course_adjustments(
    base_time_sec: float,
    distance_mi: float,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float,
) -> float:
    temp_adj = base_time_sec * (temperature_factor(temp_f) - 1.0)
    elev_adj = elevation_time_adjustment_sec(distance_mi, elev_gain_ft, elev_loss_ft)
    return base_time_sec + temp_adj + elev_adj
