"""Load multi-athlete benchmark corpora for Phase 2 evaluation."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from race_predictor.data.loader import (
    RACE_KEYWORDS,
    _distance_bucket_label,
    _parse_date,
    _parse_float,
)
from race_predictor.data.models import Run

BENCHMARK_REQUIRED_COLUMNS = {
    "athlete_id",
    "activity_id",
    "activity_date",
    "distance_mi",
    "moving_time_sec",
}

BENCHMARK_OPTIONAL_COLUMNS = {
    "name",
    "elev_gain_ft",
    "elev_loss_ft",
    "gap_pace_min_per_mi",
    "avg_hr",
    "relative_effort",
    "temp_f",
    "is_race",
}


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _is_likely_race(name: str, distance_mi: float, is_race: bool) -> bool:
    if is_race:
        return True
    if RACE_KEYWORDS.search(name):
        return True
    bucket = _distance_bucket_label(distance_mi)
    if bucket is None:
        return False
    lowered = name.lower()
    return any(kw in lowered for kw in ("marathon", "half", "5k", "10k", "race"))


def load_benchmark_corpus(csv_path: str | Path) -> list[Run]:
    """Load a multi-athlete benchmark CSV into normalized Run records."""
    path = Path(csv_path)
    runs: list[Run] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Benchmark corpus {path} is empty.")

        columns = {name.strip() for name in reader.fieldnames}
        missing = BENCHMARK_REQUIRED_COLUMNS - columns
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"Benchmark corpus {path} is missing required columns: {missing_list}"
            )

        for row in reader:
            athlete_id = row.get("athlete_id", "").strip()
            activity_id = row.get("activity_id", "").strip()
            activity_date = row.get("activity_date", "").strip()
            distance_mi = _parse_float(row.get("distance_mi"))
            moving_sec = _parse_float(row.get("moving_time_sec"))

            if not athlete_id or not activity_id or distance_mi is None or moving_sec is None:
                continue

            parsed_date = _parse_date(activity_date)
            if parsed_date is None:
                for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                    try:
                        parsed_date = datetime.strptime(activity_date, fmt)
                        break
                    except ValueError:
                        continue
            if parsed_date is None:
                continue

            if distance_mi <= 0 or moving_sec <= 0:
                continue

            name = row.get("name", "").strip() or f"Activity {activity_id}"
            is_race = _parse_bool(row.get("is_race"))

            runs.append(
                Run(
                    activity_id=activity_id,
                    date=parsed_date,
                    name=name,
                    distance_mi=distance_mi,
                    moving_time_sec=moving_sec,
                    elev_gain_ft=_parse_float(row.get("elev_gain_ft")) or 0.0,
                    elev_loss_ft=_parse_float(row.get("elev_loss_ft")) or 0.0,
                    gap_pace_min_per_mi=_parse_float(row.get("gap_pace_min_per_mi")),
                    avg_hr=_parse_float(row.get("avg_hr")),
                    relative_effort=_parse_float(row.get("relative_effort")),
                    temp_f=_parse_float(row.get("temp_f")),
                    is_likely_race=_is_likely_race(name, distance_mi, is_race),
                    athlete_id=athlete_id,
                )
            )

    runs.sort(key=lambda run: (run.athlete_id, run.date))
    return runs
