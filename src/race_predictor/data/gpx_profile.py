"""Parse GPX course files and compute elevation gain/loss in feet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gpxpy

from race_predictor.units import meters_to_feet

DEFAULT_NOISE_THRESHOLD_FT = 3.0


@dataclass(frozen=True)
class GpxElevationProfile:
    elev_gain_ft: float
    elev_loss_ft: float
    point_count: int
    source: str = "gpx"


def parse_gpx(
    path: str | Path,
    *,
    noise_threshold_ft: float = DEFAULT_NOISE_THRESHOLD_FT,
) -> GpxElevationProfile:
    """Load a GPX file and return total elevation gain and loss in feet."""
    gpx_path = Path(path)
    text = gpx_path.read_text(encoding="utf-8")
    return parse_gpx_text(text, noise_threshold_ft=noise_threshold_ft)


def parse_gpx_text(
    gpx_xml: str,
    *,
    noise_threshold_ft: float = DEFAULT_NOISE_THRESHOLD_FT,
) -> GpxElevationProfile:
    """Parse GPX XML and return total elevation gain and loss in feet."""
    gpx = gpxpy.parse(gpx_xml)
    elevations_ft = _collect_elevations_ft(gpx)
    if len(elevations_ft) < 2:
        raise ValueError("GPX must contain at least two points with elevation data")

    gain_ft, loss_ft = _sum_gain_loss(elevations_ft, noise_threshold_ft)
    return GpxElevationProfile(
        elev_gain_ft=gain_ft,
        elev_loss_ft=loss_ft,
        point_count=len(elevations_ft),
    )


def _collect_elevations_ft(gpx: gpxpy.gpx.GPX) -> list[float]:
    elevations: list[float] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.elevation is None:
                    continue
                elevations.append(meters_to_feet(float(point.elevation)))
    return elevations


def _sum_gain_loss(
    elevations_ft: list[float],
    noise_threshold_ft: float,
) -> tuple[float, float]:
    gain_ft = 0.0
    loss_ft = 0.0
    previous = elevations_ft[0]
    for current in elevations_ft[1:]:
        delta = current - previous
        if delta > noise_threshold_ft:
            gain_ft += delta
        elif delta < -noise_threshold_ft:
            loss_ft += -delta
        previous = current
    return gain_ft, loss_ft
