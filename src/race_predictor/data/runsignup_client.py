"""RunSignup REST API client for race search, detail, and event results."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RUNSIGNUP_API_BASE = "https://api.runsignup.com/rest"
REQUEST_TIMEOUT_SEC = 30
MAX_CONCURRENT_REQUESTS = 2
DEFAULT_CACHE_TTL_SEC = 3600


JsonFetcher = Callable[[str, Optional[dict[str, str]]], dict]


@dataclass(frozen=True)
class RunSignupCredentials:
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class RunSignupEvent:
    event_id: int
    name: str
    distance: float | None
    distance_units: str | None
    start_time: str | None


@dataclass(frozen=True)
class RunSignupRaceLink:
    link_id: int | None
    name: str | None
    url: str | None
    link_type: str | None


@dataclass(frozen=True)
class RunSignupRaceSummary:
    race_id: int
    name: str
    city: str | None
    state: str | None
    next_date: str | None
    last_date: str | None
    events: list[RunSignupEvent]


@dataclass(frozen=True)
class RunSignupRace:
    race_id: int
    name: str
    city: str | None
    state: str | None
    latitude: float | None
    longitude: float | None
    next_date: str | None
    last_date: str | None
    events: list[RunSignupEvent]
    race_links: tuple[RunSignupRaceLink, ...] = ()


@dataclass(frozen=True)
class RunSignupSearchResult:
    races: list[RunSignupRaceSummary]
    page: int
    results_per_page: int


@dataclass(frozen=True)
class RunSignupResult:
    registration_id: int | None
    bib: int | None
    place: int | None
    clock_time_sec: float | None
    first_name: str | None
    last_name: str | None


class RunSignupClient:
    """Thin client for RunSignup race search, detail, and finisher results."""

    def __init__(
        self,
        credentials: RunSignupCredentials | None = None,
        *,
        access_token: str | None = None,
        fetch_json: JsonFetcher | None = None,
        cache_ttl_sec: int = DEFAULT_CACHE_TTL_SEC,
    ) -> None:
        self._credentials = credentials or credentials_from_env()
        self._access_token = access_token or access_token_from_env()
        self._fetch_json = fetch_json or _fetch_json
        self._cache_ttl_sec = cache_ttl_sec
        self._cache: dict[str, tuple[float, dict]] = {}
        self._last_request_at = 0.0

    def search_races(
        self,
        *,
        name: str | None = None,
        city: str | None = None,
        state: str | None = None,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        page: int = 1,
        results_per_page: int = 25,
        today: date | None = None,
    ) -> RunSignupSearchResult:
        """Search upcoming races via GET /rest/races."""
        if page < 1:
            raise ValueError("page must be >= 1")
        if results_per_page < 1 or results_per_page > 1000:
            raise ValueError("results_per_page must be between 1 and 1000")

        reference = today or date.today()
        if start_date is None:
            min_date = reference
        else:
            parsed_start = _parse_runsignup_date(str(start_date))
            min_date = parsed_start if parsed_start is not None else reference
        if min_date < reference:
            min_date = reference

        params: dict[str, str | int] = {
            "format": "json",
            "events": "T",
            "search_start_date_only": "T",
            "page": page,
            "results_per_page": results_per_page,
            "start_date": _format_api_date(min_date),
        }
        if name:
            params["name"] = name.strip()
        if city:
            params["city"] = city.strip()
        if state:
            params["state"] = state.strip()
        if end_date is not None:
            params["end_date"] = _format_api_date(end_date)

        payload = self._get("/races", params)
        races = _filter_upcoming_summaries(
            _parse_search_results(payload),
            min_date=min_date,
        )
        return RunSignupSearchResult(
            races=races,
            page=page,
            results_per_page=results_per_page,
        )

    def get_race(
        self,
        race_id: int,
        *,
        most_recent_events_only: bool = True,
        future_events_only: bool = False,
        race_links: bool = False,
    ) -> RunSignupRace:
        params: dict[str, str] = {
            "format": "json",
            "include_event_days": "T",
        }
        if future_events_only:
            params["future_events_only"] = "T"
        else:
            params["most_recent_events_only"] = "T" if most_recent_events_only else "F"
        if race_links:
            params["race_links"] = "T"

        payload = self._get(f"/race/{race_id}", params)
        return _parse_race(payload, race_id)

    def get_event_results(
        self,
        race_id: int,
        event_id: int,
        *,
        max_place: int | None = None,
    ) -> list[RunSignupResult]:
        page_size = min(max_place or 500, 500)
        page = 1
        rows: list[RunSignupResult] = []

        while True:
            params: dict[str, str | int] = {
                "format": "json",
                "race_id": race_id,
                "event_id": event_id,
                "include_total_finishers": "T",
                "page": page,
                "results_per_page": page_size,
            }
            if max_place is not None:
                params["max_place"] = max_place

            payload = self._get(f"/race/{race_id}/results/get-results", params)
            page_rows = _parse_results(payload)
            if not page_rows:
                break
            rows.extend(page_rows)
            if len(page_rows) < page_size:
                break
            if max_place is not None and len(rows) >= max_place:
                rows = rows[:max_place]
                break
            page += 1

        return rows

    def _get(self, path: str, params: dict[str, str | int]) -> dict:
        headers: dict[str, str] = {"Accept": "application/json"}
        # Public race metadata and published finisher results are readable without
        # credentials. OAuth Bearer tokens are rejected by the legacy /rest API
        # ("Key authentication failed"), so only partner api_key/api_secret are sent.
        if self._credentials is not None:
            query = {
                "api_key": self._credentials.api_key,
                "api_secret": self._credentials.api_secret,
                **params,
            }
        else:
            query = dict(params)

        url = f"{RUNSIGNUP_API_BASE}{path}?{urlencode(query)}"
        cache_key = f"{url}|{headers.get('Authorization', '')}"
        cached = self._cache.get(cache_key)
        now = time.time()
        if cached is not None and now - cached[0] < self._cache_ttl_sec:
            return cached[1]

        self._throttle()
        try:
            payload = self._fetch_json(url, headers)
        except HTTPError as exc:
            raise RunSignupError(f"RunSignup HTTP {exc.code} for {path}") from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:
            raise RunSignupError(f"RunSignup request failed for {path}: {exc}") from exc

        if "error" in payload:
            error = payload["error"]
            if isinstance(error, dict):
                message = error.get("error_msg", str(error))
            else:
                message = str(error)
            raise RunSignupError(f"RunSignup API error: {message}")

        self._cache[cache_key] = (now, payload)
        return payload

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request_at = time.time()


class RunSignupError(Exception):
    """Raised when a RunSignup API call fails."""


def credentials_from_env() -> RunSignupCredentials | None:
    api_key = os.getenv("RUNSIGNUP_API_KEY", "").strip()
    api_secret = os.getenv("RUNSIGNUP_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None
    return RunSignupCredentials(api_key=api_key, api_secret=api_secret)


def access_token_from_env() -> str | None:
    from race_predictor.data.runsignup_oauth import access_token_from_env_or_file

    return access_token_from_env_or_file()


def has_runsignup_auth() -> bool:
    if access_token_from_env() is not None:
        return True
    if credentials_from_env() is not None:
        return True
    from race_predictor.data.runsignup_oauth import oauth_config_from_env

    return oauth_config_from_env() is not None


def _fetch_json(url: str, headers: Optional[dict[str, str]] = None) -> dict:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.load(response)


def _format_api_date(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _parse_runsignup_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(value[:19], fmt).date()
        except ValueError:
            continue
    return None


def _upcoming_race_date(summary: RunSignupRaceSummary) -> date | None:
    return _parse_runsignup_date(summary.next_date)


def _filter_upcoming_summaries(
    races: list[RunSignupRaceSummary],
    *,
    min_date: date,
) -> list[RunSignupRaceSummary]:
    upcoming: list[RunSignupRaceSummary] = []
    for race in races:
        race_date = _upcoming_race_date(race)
        if race_date is not None and race_date >= min_date:
            upcoming.append(race)
    return upcoming


def _parse_events(events_raw: Any) -> list[RunSignupEvent]:
    if not isinstance(events_raw, list):
        return []

    events: list[RunSignupEvent] = []
    for item in events_raw:
        if not isinstance(item, dict):
            continue
        event = item.get("event") if "event" in item else item
        if not isinstance(event, dict):
            continue
        event_id = _to_int(event.get("event_id"))
        if event_id is None:
            continue
        events.append(
            RunSignupEvent(
                event_id=event_id,
                name=str(event.get("name") or "").strip(),
                distance=_to_float(event.get("distance")),
                distance_units=_optional_str(event.get("distance_units")),
                start_time=_optional_str(event.get("start_time")),
            )
        )
    return events


def _parse_race_links(race: dict) -> tuple[RunSignupRaceLink, ...]:
    links: list[RunSignupRaceLink] = []
    for item in race.get("race_links") or []:
        if not isinstance(item, dict):
            continue
        link = item.get("link") if "link" in item else item
        if not isinstance(link, dict):
            continue
        links.append(
            RunSignupRaceLink(
                link_id=_to_int(link.get("link_id")),
                name=_optional_str(link.get("name") or link.get("link_name")),
                url=_optional_str(link.get("url")),
                link_type=_optional_str(link.get("link_type")),
            )
        )
    return tuple(links)


def _parse_race_summary(race: dict) -> RunSignupRaceSummary | None:
    race_id = _to_int(race.get("race_id"))
    if race_id is None:
        return None
    address = race.get("address") or {}
    return RunSignupRaceSummary(
        race_id=race_id,
        name=str(race.get("name") or "").strip(),
        city=_optional_str(address.get("city")),
        state=_optional_str(address.get("state")),
        next_date=_optional_str(race.get("next_date")),
        last_date=_optional_str(race.get("last_date")),
        events=_parse_events(race.get("events")),
    )


def _parse_search_results(payload: dict) -> list[RunSignupRaceSummary]:
    races_raw = payload.get("races") or []
    if not isinstance(races_raw, list):
        return []

    summaries: list[RunSignupRaceSummary] = []
    for item in races_raw:
        if not isinstance(item, dict):
            continue
        race = item.get("race") if "race" in item else item
        if not isinstance(race, dict):
            continue
        summary = _parse_race_summary(race)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _parse_race(payload: dict, race_id: int) -> RunSignupRace:
    race = _unwrap_race(payload)
    address = race.get("address") or {}
    events = _parse_events(race.get("events"))

    return RunSignupRace(
        race_id=race_id,
        name=str(race.get("name") or "").strip(),
        city=_optional_str(address.get("city")),
        state=_optional_str(address.get("state")),
        latitude=_to_float(address.get("latitude")),
        longitude=_to_float(address.get("longitude")),
        next_date=_optional_str(race.get("next_date")),
        last_date=_optional_str(race.get("last_date")),
        events=events,
        race_links=_parse_race_links(race),
    )


def _parse_results(payload: dict) -> list[RunSignupResult]:
    rows: list[RunSignupResult] = []

    candidates: list[Any] = []
    result_sets = payload.get("individual_results_sets")
    if isinstance(result_sets, list):
        for result_set in result_sets:
            if not isinstance(result_set, dict):
                continue
            if result_set.get("public_results") == "F":
                continue
            set_results = result_set.get("results")
            if isinstance(set_results, list):
                candidates.extend(set_results)
    elif isinstance(payload.get("results"), list):
        candidates = payload["results"]
    elif isinstance(payload.get("result"), list):
        candidates = payload["result"]
    else:
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if any(key in value[0] for key in ("clock_time", "chip_time", "place")):
                    candidates = value
                    break

    for item in candidates:
        if not isinstance(item, dict):
            continue
        result = item.get("result") if "result" in item else item
        if not isinstance(result, dict):
            continue
        clock = _parse_clock_time(
            result.get("clock_time")
            or result.get("chip_time")
            or result.get("gun_time")
        )
        registration_id = _to_int(result.get("registration_id"))
        if registration_id is None:
            registration_id = _to_int(result.get("result_id"))
        rows.append(
            RunSignupResult(
                registration_id=registration_id,
                bib=_to_int(result.get("bib")),
                place=_to_int(result.get("place")),
                clock_time_sec=clock,
                first_name=_optional_str(result.get("first_name")),
                last_name=_optional_str(result.get("last_name")),
            )
        )
    return rows


def _unwrap_race(payload: dict) -> dict:
    if "race" in payload and isinstance(payload["race"], dict):
        return payload["race"]
    return payload


def _parse_clock_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return float(text)
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            hours_i = int(hours)
            minutes_i = int(minutes)
            seconds_f = float(seconds)
            # RunSignup sometimes omits the hour field for sub-hour finishes
            # (e.g. "18:30:00" meaning 18 minutes, not 18 hours).
            if hours_i >= 4 and hours_i < 60 and minutes_i < 60:
                return hours_i * 60 + minutes_i + seconds_f
            return hours_i * 3600 + minutes_i * 60 + seconds_f
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 1 and ":" not in text and text.replace(".", "", 1).isdigit():
            return float(text)
    except ValueError:
        return None
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
