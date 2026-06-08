"""Build and validate a multi-athlete training corpus from RunSignup results."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from race_predictor.constants import DISTANCE_BUCKETS_MI, RACE_DISTANCES_MI
from race_predictor.data.models import Run
from race_predictor.data.runsignup_client import (
    RunSignupClient,
    RunSignupError,
    RunSignupEvent,
    RunSignupRace,
    RunSignupResult,
)
from race_predictor.data.weather import fetch_race_day_weather
from race_predictor.units import meters_to_miles

DEFAULT_CATALOG_PATH = Path("catalog/races.json")
DEFAULT_CORPUS_PATH = Path("benchmarks/runsignup_corpus.csv")
DEFAULT_COVERAGE_REPORT_PATH = Path("reports/corpus_coverage.json")

CORPUS_CSV_COLUMNS = [
    "athlete_id",
    "activity_id",
    "activity_date",
    "name",
    "distance_mi",
    "moving_time_sec",
    "elev_gain_ft",
    "elev_loss_ft",
    "gap_pace_min_per_mi",
    "avg_hr",
    "relative_effort",
    "temp_f",
    "is_race",
]

MIN_FINISHERS_PER_DISTANCE = 50
MIN_RACES_PER_DISTANCE = 2
MAX_FINISHERS_PER_EVENT = 1000


@dataclass(frozen=True)
class CatalogRace:
    runsignup_race_id: int
    name: str
    distance_label: str
    course_type: str
    elev_gain_ft: float
    elev_loss_ft: float
    event_name_contains: str | None = None
    event_name_excludes: str | None = None
    priority: str = "required"


@dataclass
class RaceSyncStats:
    catalog_entry: str
    race_id: int
    distance_label: str
    course_type: str
    event_id: int | None = None
    event_name: str | None = None
    finishers: int = 0
    skipped: int = 0
    error: str | None = None


@dataclass
class CoverageReport:
    total_finishers: int = 0
    total_races_synced: int = 0
    by_distance: dict[str, dict[str, int]] = field(default_factory=dict)
    by_course_type: dict[str, int] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    race_stats: list[dict[str, object]] = field(default_factory=list)
    passed: bool = False


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> list[CatalogRace]:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = payload.get("races") or []
    races: list[CatalogRace] = []
    for entry in entries:
        races.append(
            CatalogRace(
                runsignup_race_id=int(entry["runsignup_race_id"]),
                name=str(entry["name"]),
                distance_label=str(entry["distance_label"]),
                course_type=str(entry["course_type"]),
                elev_gain_ft=float(entry["elev_gain_ft"]),
                elev_loss_ft=float(entry["elev_loss_ft"]),
                event_name_contains=entry.get("event_name_contains"),
                event_name_excludes=entry.get("event_name_excludes"),
                priority=str(entry.get("priority", "required")),
            )
        )
    return races


def sync_corpus(
    client: RunSignupClient,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    output_path: str | Path = DEFAULT_CORPUS_PATH,
    coverage_report_path: str | Path = DEFAULT_COVERAGE_REPORT_PATH,
    max_finishers_per_event: int = MAX_FINISHERS_PER_EVENT,
) -> tuple[list[Run], CoverageReport]:
    """Fetch curated RunSignup races and write a benchmark-style corpus CSV."""
    catalog = load_catalog(catalog_path)
    runs: list[Run] = []
    race_stats: list[RaceSyncStats] = []

    for entry in catalog:
        stats = RaceSyncStats(
            catalog_entry=f"{entry.name} ({entry.distance_label})",
            race_id=entry.runsignup_race_id,
            distance_label=entry.distance_label,
            course_type=entry.course_type,
        )
        try:
            race = client.get_race(entry.runsignup_race_id)
            event = select_event(race.events, entry)
            if event is None:
                stats.error = f"No matching event for {entry.distance_label}"
                race_stats.append(stats)
                continue

            stats.event_id = event.event_id
            stats.event_name = event.name
            race_date = parse_race_date(race)
            temp_f = fetch_race_temp(race, race_date)
            results = client.get_event_results(
                entry.runsignup_race_id,
                event.event_id,
                max_place=max_finishers_per_event,
            )

            distance_mi = event_distance_mi(event, entry.distance_label)
            event_runs, skipped = results_to_runs(
                results,
                race_id=entry.runsignup_race_id,
                event_id=event.event_id,
                race_name=entry.name,
                event_name=event.name,
                race_date=race_date,
                distance_mi=distance_mi,
                elev_gain_ft=entry.elev_gain_ft,
                elev_loss_ft=entry.elev_loss_ft,
                temp_f=temp_f,
                course_type=entry.course_type,
            )
            runs.extend(event_runs)
            stats.finishers = len(event_runs)
            stats.skipped = skipped
        except RunSignupError as exc:
            stats.error = str(exc)
        race_stats.append(stats)

    report = build_coverage_report(runs, race_stats)
    write_corpus_csv(runs, output_path)
    write_coverage_report(report, coverage_report_path)
    return runs, report


def select_event(events: list[RunSignupEvent], entry: CatalogRace) -> RunSignupEvent | None:
    candidates = list(events)
    if entry.event_name_contains:
        needle = entry.event_name_contains.lower()
        candidates = [event for event in candidates if needle in event.name.lower()]
    if entry.event_name_excludes:
        exclude = entry.event_name_excludes.lower()
        candidates = [
            event for event in candidates if exclude not in event.name.lower()
        ]

    if not candidates:
        candidates = list(events)

    target_mi = RACE_DISTANCES_MI[entry.distance_label]
    scored: list[tuple[float, RunSignupEvent]] = []
    for event in candidates:
        distance_mi = raw_event_distance_mi(event)
        if distance_mi is None:
            continue
        scored.append((abs(distance_mi - target_mi), event))

    if scored:
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    if candidates:
        return candidates[0]
    return None


def raw_event_distance_mi(event: RunSignupEvent) -> float | None:
    if event.distance is None:
        return None
    units = (event.distance_units or "M").upper()
    if units == "K":
        return meters_to_miles(event.distance * 1000.0)
    if units in {"M", "MI", "MILE", "MILES"}:
        return float(event.distance)
    if units in {"METER", "METERS", "m"}:
        return meters_to_miles(event.distance)
    return float(event.distance)


def event_distance_mi(event: RunSignupEvent, distance_label: str) -> float:
    measured = raw_event_distance_mi(event)
    if measured is not None:
        lo, hi = DISTANCE_BUCKETS_MI[distance_label]
        if lo <= measured <= hi:
            return measured
    return RACE_DISTANCES_MI[distance_label]


def parse_race_date(race: RunSignupRace) -> datetime:
    for candidate in (race.last_date, race.next_date):
        if not candidate:
            continue
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(candidate[:19], fmt)
            except ValueError:
                continue
    return datetime.now()


def fetch_race_temp(race: RunSignupRace, race_date: datetime) -> float | None:
    if race.latitude is None or race.longitude is None:
        return None
    weather = fetch_race_day_weather(
        race.latitude,
        race.longitude,
        race_date.date(),
    )
    return weather.temp_f


def results_to_runs(
    results: Iterable[RunSignupResult],
    *,
    race_id: int,
    event_id: int,
    race_name: str,
    event_name: str,
    race_date: datetime,
    distance_mi: float,
    elev_gain_ft: float,
    elev_loss_ft: float,
    temp_f: float | None,
    course_type: str,
) -> tuple[list[Run], int]:
    runs: list[Run] = []
    skipped = 0
    for result in results:
        if result.clock_time_sec is None or result.clock_time_sec <= 0:
            skipped += 1
            continue

        athlete_id = (
            str(result.registration_id)
            if result.registration_id is not None
            else f"bib-{result.bib or skipped}"
        )
        activity_id = f"{race_id}:{event_id}:{athlete_id}"
        display_name = " ".join(
            part
            for part in (result.first_name, result.last_name)
            if part
        ).strip()
        if not display_name:
            display_name = f"{race_name} - {event_name}"

        runs.append(
            Run(
                activity_id=activity_id,
                date=race_date,
                name=f"{display_name} ({course_type})",
                distance_mi=distance_mi,
                moving_time_sec=result.clock_time_sec,
                elev_gain_ft=elev_gain_ft,
                elev_loss_ft=elev_loss_ft,
                gap_pace_min_per_mi=None,
                avg_hr=None,
                relative_effort=None,
                temp_f=temp_f,
                is_likely_race=True,
                athlete_id=athlete_id,
            )
        )
    return runs, skipped


def build_coverage_report(
    runs: list[Run],
    race_stats: list[RaceSyncStats],
) -> CoverageReport:
    by_distance: dict[str, dict[str, int]] = {
        label: {"finishers": 0, "races": 0} for label in RACE_DISTANCES_MI
    }
    by_course_type: dict[str, int] = {}

    for stats in race_stats:
        if stats.error or stats.finishers == 0:
            continue
        by_distance[stats.distance_label]["finishers"] += stats.finishers
        by_distance[stats.distance_label]["races"] += 1
        by_course_type[stats.course_type] = (
            by_course_type.get(stats.course_type, 0) + stats.finishers
        )

    gaps: list[str] = []
    for label, bucket in by_distance.items():
        if bucket["races"] < MIN_RACES_PER_DISTANCE:
            gaps.append(
                f"{label}: only {bucket['races']} races synced "
                f"(need {MIN_RACES_PER_DISTANCE})"
            )
        if bucket["finishers"] < MIN_FINISHERS_PER_DISTANCE:
            gaps.append(
                f"{label}: only {bucket['finishers']} finishers "
                f"(need {MIN_FINISHERS_PER_DISTANCE})"
            )

    for course_type in {"downhill", "uphill", "flat", "rolling"}:
        if by_course_type.get(course_type, 0) == 0:
            gaps.append(f"course_type '{course_type}': no finishers collected")

    return CoverageReport(
        total_finishers=len(runs),
        total_races_synced=sum(1 for stats in race_stats if stats.finishers > 0),
        by_distance=by_distance,
        by_course_type=by_course_type,
        gaps=gaps,
        race_stats=[asdict(stats) for stats in race_stats],
        passed=len(gaps) == 0 and len(runs) > 0,
    )


def write_corpus_csv(runs: list[Run], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_CSV_COLUMNS)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "athlete_id": run.athlete_id,
                    "activity_id": run.activity_id,
                    "activity_date": run.date.strftime("%Y-%m-%d"),
                    "name": run.name,
                    "distance_mi": f"{run.distance_mi:.5f}",
                    "moving_time_sec": f"{run.moving_time_sec:.1f}",
                    "elev_gain_ft": f"{run.elev_gain_ft:.1f}",
                    "elev_loss_ft": f"{run.elev_loss_ft:.1f}",
                    "gap_pace_min_per_mi": "",
                    "avg_hr": "",
                    "relative_effort": "",
                    "temp_f": "" if run.temp_f is None else f"{run.temp_f:.1f}",
                    "is_race": "true",
                }
            )


def write_coverage_report(report: CoverageReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_finishers": report.total_finishers,
                "total_races_synced": report.total_races_synced,
                "by_distance": report.by_distance,
                "by_course_type": report.by_course_type,
                "gaps": report.gaps,
                "passed": report.passed,
                "race_stats": report.race_stats,
            },
            handle,
            indent=2,
        )


def format_coverage_summary(report: CoverageReport) -> str:
    lines = [
        f"Total finishers: {report.total_finishers}",
        f"Races synced: {report.total_races_synced}",
        "",
        "By distance:",
    ]
    for label in RACE_DISTANCES_MI:
        bucket = report.by_distance.get(label, {"finishers": 0, "races": 0})
        lines.append(
            f"  {label:<10} {bucket['finishers']:>5} finishers  "
            f"{bucket['races']:>2} races"
        )
    lines.append("")
    lines.append("By course type:")
    for course_type, count in sorted(report.by_course_type.items()):
        lines.append(f"  {course_type:<10} {count:>5} finishers")
    if report.gaps:
        lines.append("")
        lines.append("Coverage gaps:")
        for gap in report.gaps:
            lines.append(f"  - {gap}")
    lines.append("")
    lines.append(f"Coverage check: {'PASSED' if report.passed else 'FAILED'}")
    return "\n".join(lines)
