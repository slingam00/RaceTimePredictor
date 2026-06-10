"""Tests for RunSignup API client."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from race_predictor.data.runsignup_client import (
    RunSignupClient,
    RunSignupCredentials,
    _parse_clock_time,
    _parse_race,
    _parse_results,
    _parse_search_results,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_clock_time_formats():
    assert _parse_clock_time("00:18:30") == 18 * 60 + 30
    assert _parse_clock_time("18:30:00") == 18 * 60 + 30
    assert _parse_clock_time("1:05:30") == 3930
    assert _parse_clock_time("3:45:00") == 3 * 3600 + 45 * 60
    assert _parse_clock_time("25:00") == 1500
    assert _parse_clock_time("20:58.6") == 20 * 60 + 58.6
    assert _parse_clock_time(1500) == 1500.0


def test_parse_race_fixture():
    payload = json.loads((FIXTURES / "runsignup_race_74589.json").read_text())
    race = _parse_race(payload, 74589)
    assert race.race_id == 74589
    assert race.name.startswith("Cambridge")
    assert len(race.events) == 2
    assert race.events[0].event_id == 101


def test_parse_results_fixture():
    payload = json.loads(
        (FIXTURES / "runsignup_results_74589_101.json").read_text()
    )
    results = _parse_results(payload)
    assert len(results) == 3
    assert results[0].registration_id == 9001
    assert results[0].clock_time_sec == 18 * 60 + 30


def test_parse_results_individual_sets_fixture():
    payload = json.loads(
        (FIXTURES / "runsignup_results_individual_sets.json").read_text()
    )
    results = _parse_results(payload)
    assert len(results) == 2
    assert results[0].registration_id == 31258816
    assert results[0].clock_time_sec == 20 * 60 + 58.6


def test_parse_search_results_fixture():
    payload = json.loads((FIXTURES / "runsignup_search_marathon.json").read_text())
    races = _parse_search_results(payload)
    assert len(races) == 2
    assert races[0].race_id == 146508
    assert races[0].city == "Port Huron"
    assert races[0].state == "MI"
    assert len(races[0].events) == 2
    assert races[1].race_id == 72232
    assert races[1].events[0].name == "Full Marathon"


def test_parse_race_future_links_fixture():
    payload = json.loads((FIXTURES / "runsignup_race_future_links.json").read_text())
    race = _parse_race(payload, 146508)
    assert race.next_date == "08/09/2026"
    assert len(race.events) == 1
    assert len(race.race_links) == 1
    assert race.race_links[0].url == "https://example.com/course.gpx"
    assert race.race_links[0].link_type == "gpx"


def test_search_races_builds_query_and_parses():
    calls: list[str] = []

    def fetch(url: str, headers: dict | None = None) -> dict:
        calls.append(url)
        return json.loads((FIXTURES / "runsignup_search_marathon.json").read_text())

    client = RunSignupClient(
        credentials=RunSignupCredentials(api_key="k", api_secret="s"),
        fetch_json=fetch,
    )
    result = client.search_races(
        name="marathon",
        city="Port Huron",
        state="MI",
        start_date="2026-01-01",
        end_date="2026-12-31",
        page=2,
        results_per_page=10,
        today=date(2026, 1, 1),
    )
    assert result.page == 2
    assert result.results_per_page == 10
    assert len(result.races) == 2
    assert result.races[0].name.startswith("Bridge to Brew")
    assert "name=marathon" in calls[0]
    assert "city=Port+Huron" in calls[0]
    assert "state=MI" in calls[0]
    assert "start_date=2026-01-01" in calls[0]
    assert "end_date=2026-12-31" in calls[0]
    assert "events=T" in calls[0]
    assert "search_start_date_only=T" in calls[0]
    assert "page=2" in calls[0]
    assert "results_per_page=10" in calls[0]


def test_search_races_filters_past_next_date():
    payload = {
        "races": [
            {
                "race": {
                    "race_id": 1,
                    "name": "Past Marathon",
                    "next_date": "01/15/2025",
                    "last_date": "01/15/2025",
                    "address": {"city": "A", "state": "MA"},
                    "events": [],
                }
            },
            {
                "race": {
                    "race_id": 2,
                    "name": "Future Marathon",
                    "next_date": "09/14/2026",
                    "last_date": "09/14/2025",
                    "address": {"city": "B", "state": "MA"},
                    "events": [],
                }
            },
        ]
    }

    def fetch(url: str, headers: dict | None = None) -> dict:
        return payload

    client = RunSignupClient(
        credentials=RunSignupCredentials(api_key="k", api_secret="s"),
        fetch_json=fetch,
    )
    result = client.search_races(
        name="marathon",
        start_date="2026-06-08",
        today=__import__("datetime").date(2026, 6, 8),
    )
    assert len(result.races) == 1
    assert result.races[0].race_id == 2


def test_search_races_filters_beyond_max_date():
    payload = {
        "races": [
            {
                "race": {
                    "race_id": 1,
                    "name": "Near Marathon",
                    "next_date": "07/15/2026",
                    "address": {"city": "A", "state": "MA"},
                    "events": [],
                }
            },
            {
                "race": {
                    "race_id": 2,
                    "name": "Far Marathon",
                    "next_date": "09/14/2026",
                    "address": {"city": "B", "state": "MA"},
                    "events": [],
                }
            },
        ]
    }
    calls: list[str] = []

    def fetch(url: str, headers: dict | None = None) -> dict:
        calls.append(url)
        return payload

    client = RunSignupClient(
        credentials=RunSignupCredentials(api_key="k", api_secret="s"),
        fetch_json=fetch,
    )
    result = client.search_races(
        start_date="2026-06-08",
        max_date="2026-08-24",
        today=date(2026, 6, 8),
    )
    assert len(result.races) == 1
    assert result.races[0].race_id == 1
    assert "end_date=2026-08-24" in calls[0]


def test_get_race_future_events_only_passes_params():
    calls: list[str] = []

    def fetch(url: str, headers: dict | None = None) -> dict:
        calls.append(url)
        return json.loads((FIXTURES / "runsignup_race_future_links.json").read_text())

    client = RunSignupClient(
        credentials=RunSignupCredentials(api_key="k", api_secret="s"),
        fetch_json=fetch,
    )
    race = client.get_race(146508, future_events_only=True, race_links=True)
    assert race.events[0].event_id == 1140658
    assert "future_events_only=T" in calls[0]
    assert "race_links=T" in calls[0]
    assert "most_recent_events_only" not in calls[0]


def test_client_uses_cache():
    calls: list[str] = []

    def fetch(url: str, headers: dict | None = None) -> dict:
        calls.append(url)
        return json.loads((FIXTURES / "runsignup_race_74589.json").read_text())

    client = RunSignupClient(
        credentials=RunSignupCredentials(api_key="k", api_secret="s"),
        fetch_json=fetch,
        cache_ttl_sec=60,
    )
    race1 = client.get_race(74589)
    race2 = client.get_race(74589)
    assert race1.name == race2.name
    assert len(calls) == 1
