"""Tests for RunSignup API client."""

from __future__ import annotations

import json
from pathlib import Path

from race_predictor.data.runsignup_client import (
    RunSignupClient,
    RunSignupCredentials,
    _parse_clock_time,
    _parse_race,
    _parse_results,
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
