"""Enrich upcoming RunSignup races with elevation, weather, and distance metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

from race_predictor.constants import DISTANCE_BUCKETS_MI, RACE_DISTANCES_MI
from race_predictor.data.gpx_profile import parse_gpx, parse_gpx_text
from race_predictor.data.runsignup_client import RunSignupClient, RunSignupEvent, RunSignupRace
from race_predictor.data.runsignup_corpus import parse_race_date, raw_event_distance_mi
from race_predictor.data.weather import fetch_race_day_weather, geocode_us_city

DEFAULT_OVERRIDES_PATH = Path("catalog/overrides.json")
DEFAULT_CACHE_DIR = Path("catalog/cache")
DEFAULT_GPX_DIR = Path("catalog/gpx")
DEFAULT_CACHE_TTL = timedelta(hours=24)
REQUEST_TIMEOUT_SEC = 15

TextFetcher = Callable[[str], str]


@dataclass(frozen=True)
class RaceOverride:
    runsignup_race_id: int
    name: str
    course_type: str
    elev_gain_ft: float
    elev_loss_ft: float


@dataclass(frozen=True)
class EnrichedRaceEvent:
    event_id: int
    name: str
    distance_label: str
    distance_mi: float


@dataclass
class EnrichedRace:
    race_id: int
    name: str
    city: str | None
    state: str | None
    race_date: date
    elev_gain_ft: float | None
    elev_loss_ft: float | None
    elev_source: str | None
    temp_f: float | None
    weather_source: str | None
    offered_events: list[EnrichedRaceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_overrides(path: str | Path = DEFAULT_OVERRIDES_PATH) -> dict[int, RaceOverride]:
    override_path = Path(path)
    if not override_path.is_file():
        return {}

    payload = json.loads(override_path.read_text(encoding="utf-8"))
    overrides: dict[int, RaceOverride] = {}
    for entry in payload.get("races") or []:
        race_id = int(entry["runsignup_race_id"])
        overrides[race_id] = RaceOverride(
            runsignup_race_id=race_id,
            name=str(entry.get("name") or ""),
            course_type=str(entry.get("course_type") or "rolling"),
            elev_gain_ft=float(entry["elev_gain_ft"]),
            elev_loss_ft=float(entry["elev_loss_ft"]),
        )
    return overrides


def enrich_race(
    client: RunSignupClient,
    race_id: int,
    *,
    as_of: date | None = None,
    overrides_path: str | Path = DEFAULT_OVERRIDES_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    gpx_dir: str | Path = DEFAULT_GPX_DIR,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
    fetch_text: TextFetcher | None = None,
    today: date | None = None,
) -> EnrichedRace:
    """Fetch and enrich an upcoming race with elevation and weather."""
    cache_path = Path(cache_dir) / f"{race_id}.json"
    cached = _read_cache(cache_path, cache_ttl, today=today)
    if cached is not None:
        return cached

    race = client.get_race(race_id, future_events_only=True, race_links=True)
    overrides = load_overrides(overrides_path)
    race_date = _race_date(race, as_of=as_of)
    warnings: list[str] = []

    elev_gain_ft, elev_loss_ft, elev_source, elev_warnings = _resolve_elevation(
        race,
        overrides=overrides,
        gpx_dir=Path(gpx_dir),
        fetch_text=fetch_text,
    )
    warnings.extend(elev_warnings)

    temp_f, weather_source, weather_warnings = _resolve_weather(
        race,
        race_date,
        today=today,
    )
    warnings.extend(weather_warnings)

    offered_events = _map_offered_events(race.events)
    if not offered_events:
        warnings.append("No standard-distance events (5K, 10K, Half, Marathon) found.")

    enriched = EnrichedRace(
        race_id=race.race_id,
        name=race.name,
        city=race.city,
        state=race.state,
        race_date=race_date,
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        elev_source=elev_source,
        temp_f=temp_f,
        weather_source=weather_source,
        offered_events=offered_events,
        warnings=warnings,
    )
    _write_cache(cache_path, enriched)
    return enriched


def _race_date(race: RunSignupRace, *, as_of: date | None) -> date:
    if as_of is not None:
        return as_of
    for candidate in (race.next_date, race.last_date):
        if not candidate:
            continue
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(candidate[:19], fmt).date()
            except ValueError:
                continue
    return parse_race_date(race).date()


def _map_offered_events(events: list[RunSignupEvent]) -> list[EnrichedRaceEvent]:
    offered: list[EnrichedRaceEvent] = []
    seen_labels: set[str] = set()
    for event in events:
        if "virtual" in event.name.lower():
            continue
        label = _distance_label_for_event(event)
        if label is None or label in seen_labels:
            continue
        distance_mi = raw_event_distance_mi(event)
        offered.append(
            EnrichedRaceEvent(
                event_id=event.event_id,
                name=event.name,
                distance_label=label,
                distance_mi=distance_mi if distance_mi is not None else RACE_DISTANCES_MI[label],
            )
        )
        seen_labels.add(label)
    return offered


def _distance_label_for_event(event: RunSignupEvent) -> str | None:
    distance_mi = raw_event_distance_mi(event)
    if distance_mi is not None:
        for label, (lo, hi) in DISTANCE_BUCKETS_MI.items():
            if lo <= distance_mi <= hi:
                return label
    return _distance_label_from_name(event.name)


def _distance_label_from_name(name: str) -> str | None:
    text = name.lower()
    if "virtual" in text:
        return None
    if "marathon" in text and "half" not in text:
        return "Marathon"
    if "half" in text:
        return "Half"
    if "10k" in text or "10 k" in text:
        return "10K"
    if "5k" in text or "5 k" in text:
        return "5K"
    return None


def _resolve_elevation(
    race: RunSignupRace,
    *,
    overrides: dict[int, RaceOverride],
    gpx_dir: Path,
    fetch_text: TextFetcher | None,
) -> tuple[float | None, float | None, str | None, list[str]]:
    warnings: list[str] = []

    bundled = gpx_dir / f"{race.race_id}.gpx"
    if bundled.is_file():
        profile = parse_gpx(bundled)
        return profile.elev_gain_ft, profile.elev_loss_ft, "gpx", warnings

    for link in race.race_links:
        url = link.url or ""
        if not _looks_like_gpx_link(url, link.name, link.link_type):
            continue
        profile = _fetch_gpx_profile(url, fetch_text=fetch_text)
        if profile is not None:
            return profile.elev_gain_ft, profile.elev_loss_ft, "gpx_url", warnings

    override = overrides.get(race.race_id)
    if override is not None:
        return override.elev_gain_ft, override.elev_loss_ft, "override", warnings

    warnings.append("Elevation unknown — enter gain/loss manually.")
    return None, None, None, warnings


def _resolve_weather(
    race: RunSignupRace,
    race_date: date,
    *,
    today: date | None,
) -> tuple[float | None, str | None, list[str]]:
    warnings: list[str] = []
    latitude = race.latitude
    longitude = race.longitude
    if latitude is None or longitude is None:
        coords = geocode_us_city(race.city, race.state)
        if coords is None:
            warnings.append("Could not geocode race location for weather lookup.")
            return None, None, warnings
        latitude, longitude = coords

    weather = fetch_race_day_weather(
        latitude,
        longitude,
        race_date,
        today=today,
    )
    if weather.source == "model_default":
        warnings.append("Weather unavailable — using model default temperature.")
        return weather.temp_f, weather.source, warnings
    return weather.temp_f, weather.source, warnings


def _looks_like_gpx_link(url: str, name: str | None, link_type: str | None) -> bool:
    url_lower = url.lower()
    if url_lower.endswith(".gpx"):
        return True
    if link_type and "gpx" in link_type.lower():
        return True
    label = (name or "").lower()
    return any(token in label for token in ("gpx", "course map", "course file"))


def _fetch_gpx_profile(url: str, *, fetch_text: TextFetcher | None):
    loader = fetch_text or _fetch_text
    try:
        gpx_xml = loader(url)
    except (URLError, OSError, ValueError):
        return None
    try:
        return parse_gpx_text(gpx_xml)
    except ValueError:
        return None


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=REQUEST_TIMEOUT_SEC) as response:
        return response.read().decode("utf-8", errors="replace")


def _read_cache(
    path: Path,
    cache_ttl: timedelta,
    *,
    today: date | None,
) -> EnrichedRace | None:
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    cached_at_raw = payload.get("cached_at")
    if not cached_at_raw:
        return None
    try:
        cached_at = datetime.fromisoformat(str(cached_at_raw))
    except ValueError:
        return None

    reference = today or date.today()
    if reference - cached_at.date() > cache_ttl:
        return None

    return _enriched_from_dict(payload)


def _write_cache(path: Path, enriched: EnrichedRace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(enriched)
    payload["race_date"] = enriched.race_date.isoformat()
    payload["cached_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _enriched_from_dict(payload: dict) -> EnrichedRace:
    race_date = date.fromisoformat(str(payload["race_date"]))
    offered_events = [
        EnrichedRaceEvent(
            event_id=int(item["event_id"]),
            name=str(item["name"]),
            distance_label=str(item["distance_label"]),
            distance_mi=float(item["distance_mi"]),
        )
        for item in payload.get("offered_events") or []
    ]
    return EnrichedRace(
        race_id=int(payload["race_id"]),
        name=str(payload["name"]),
        city=payload.get("city"),
        state=payload.get("state"),
        race_date=race_date,
        elev_gain_ft=_optional_float(payload.get("elev_gain_ft")),
        elev_loss_ft=_optional_float(payload.get("elev_loss_ft")),
        elev_source=payload.get("elev_source"),
        temp_f=_optional_float(payload.get("temp_f")),
        weather_source=payload.get("weather_source"),
        offered_events=offered_events,
        warnings=[str(item) for item in payload.get("warnings") or []],
    )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
