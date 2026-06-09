"""Tests for upcoming race enrichment."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from race_predictor.data.race_enrichment import enrich_race, load_overrides
from race_predictor.data.runsignup_client import (
    RunSignupClient,
    RunSignupCredentials,
    RunSignupEvent,
    RunSignupRace,
    RunSignupRaceLink,
)
from race_predictor.data.weather import RaceDayWeather

FIXTURES = Path(__file__).parent / "fixtures"
GPX_FIXTURE = FIXTURES / "simple_hill.gpx"


class MockRunSignupClient(RunSignupClient):
    def __init__(self, race: RunSignupRace) -> None:
        super().__init__(credentials=RunSignupCredentials(api_key="k", api_secret="s"))
        self._race = race

    def get_race(
        self,
        race_id: int,
        *,
        most_recent_events_only: bool = True,
        future_events_only: bool = False,
        race_links: bool = False,
    ) -> RunSignupRace:
        return self._race


def _future_race(*, race_id: int = 9001, links: tuple[RunSignupRaceLink, ...] = ()) -> RunSignupRace:
    return RunSignupRace(
        race_id=race_id,
        name="Sample Spring 5K and Half",
        city="Cambridge",
        state="MA",
        latitude=None,
        longitude=None,
        next_date="2026-09-14",
        last_date="2025-09-14",
        events=[
            RunSignupEvent(201, "Sample 5K", 3.1, "M", "08:00:00"),
            RunSignupEvent(202, "Sample Half Marathon", 13.1, "M", "07:00:00"),
            RunSignupEvent(203, "Virtual 5K", 3.1, "M", "08:00:00"),
        ],
        race_links=links,
    )


def test_load_overrides_reads_catalog(tmp_path):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "races": [
                    {
                        "runsignup_race_id": 42,
                        "name": "Test Race",
                        "course_type": "flat",
                        "elev_gain_ft": 80,
                        "elev_loss_ft": 80,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    overrides = load_overrides(overrides_path)
    assert overrides[42].elev_gain_ft == 80


def test_enrich_race_uses_bundled_gpx(tmp_path, monkeypatch):
    race = _future_race()
    client = MockRunSignupClient(race)

    gpx_dir = tmp_path / "gpx"
    gpx_dir.mkdir()
    (gpx_dir / "9001.gpx").write_text(GPX_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.fetch_race_day_weather",
        lambda *args, **kwargs: RaceDayWeather(temp_f=68.0, source="forecast"),
    )
    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.geocode_us_city",
        lambda city, state: (42.36, -71.06),
    )

    enriched = enrich_race(
        client,
        9001,
        overrides_path=tmp_path / "missing.json",
        cache_dir=tmp_path / "cache",
        gpx_dir=gpx_dir,
        today=date(2026, 6, 1),
    )
    assert enriched.elev_source == "gpx"
    assert enriched.elev_gain_ft == pytest.approx(100.0, rel=1e-3)
    assert enriched.temp_f == pytest.approx(68.0)
    assert enriched.weather_source == "forecast"
    assert len(enriched.offered_events) == 2
    assert enriched.offered_events[0].distance_label == "5K"


def test_enrich_race_uses_override_when_no_gpx(tmp_path, monkeypatch):
    race = _future_race(race_id=72232)
    client = MockRunSignupClient(race)

    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(
        (Path("catalog/overrides.json")).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.fetch_race_day_weather",
        lambda *args, **kwargs: RaceDayWeather(temp_f=75.0, source="typical"),
    )
    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.geocode_us_city",
        lambda city, state: (26.71, -80.05),
    )

    enriched = enrich_race(
        client,
        72232,
        overrides_path=overrides_path,
        cache_dir=tmp_path / "cache",
        gpx_dir=tmp_path / "gpx",
        today=date(2026, 6, 1),
    )
    assert enriched.elev_source == "override"
    assert enriched.elev_gain_ft == 100
    assert enriched.elev_loss_ft == 100
    assert not any("Elevation unknown" in warning for warning in enriched.warnings)


def test_enrich_race_prefers_bundled_gpx_over_override(tmp_path, monkeypatch):
    race = _future_race(race_id=72232)
    client = MockRunSignupClient(race)

    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(
        (Path("catalog/overrides.json")).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    gpx_dir = tmp_path / "gpx"
    gpx_dir.mkdir()
    (gpx_dir / "72232.gpx").write_text(GPX_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.fetch_race_day_weather",
        lambda *args, **kwargs: RaceDayWeather(temp_f=70.0, source="forecast"),
    )
    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.geocode_us_city",
        lambda city, state: (26.71, -80.05),
    )

    enriched = enrich_race(
        client,
        72232,
        overrides_path=overrides_path,
        cache_dir=tmp_path / "cache",
        gpx_dir=gpx_dir,
        today=date(2026, 6, 1),
    )
    assert enriched.elev_source == "gpx"
    assert enriched.elev_gain_ft == pytest.approx(100.0, rel=1e-3)


def test_enrich_race_reads_fresh_cache_without_refetch(tmp_path, monkeypatch):
    race = _future_race(race_id=5555)
    client = MockRunSignupClient(race)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "5555.json").write_text(
        json.dumps(
            {
                "race_id": 5555,
                "name": "Cached Race",
                "city": "Boston",
                "state": "MA",
                "race_date": "2026-10-10",
                "elev_gain_ft": 200,
                "elev_loss_ft": 180,
                "elev_source": "cache",
                "temp_f": 55,
                "weather_source": "archive",
                "offered_events": [
                    {
                        "event_id": 1,
                        "name": "5K",
                        "distance_label": "5K",
                        "distance_mi": 3.10686,
                    }
                ],
                "warnings": [],
                "cached_at": "2026-06-07T10:00:00",
            }
        ),
        encoding="utf-8",
    )

    def fail_get_race(*args, **kwargs):
        raise AssertionError("get_race should not be called when cache is fresh")

    monkeypatch.setattr(client, "get_race", fail_get_race)

    enriched = enrich_race(
        client,
        5555,
        cache_dir=cache_dir,
        overrides_path=tmp_path / "missing.json",
        gpx_dir=tmp_path / "gpx",
        today=date(2026, 6, 7),
    )
    assert enriched.name == "Cached Race"
    assert enriched.elev_source == "cache"
    assert enriched.temp_f == 55


def test_distance_label_from_name_handles_common_event_names():
    from race_predictor.data.race_enrichment import _distance_label_from_name

    assert _distance_label_from_name("Half Marathon") == "Half"
    assert _distance_label_from_name("5K Run") == "5K"
    assert _distance_label_from_name("Full Marathon") == "Marathon"
    assert _distance_label_from_name("Virtual 5K") is None


def test_enrich_race_warns_when_elevation_missing(tmp_path, monkeypatch):
    race = _future_race(race_id=7777)
    client = MockRunSignupClient(race)

    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.fetch_race_day_weather",
        lambda *args, **kwargs: RaceDayWeather(temp_f=60.0, source="forecast"),
    )
    monkeypatch.setattr(
        "race_predictor.data.race_enrichment.geocode_us_city",
        lambda city, state: (42.36, -71.06),
    )

    enriched = enrich_race(
        client,
        7777,
        overrides_path=tmp_path / "missing.json",
        cache_dir=tmp_path / "cache",
        gpx_dir=tmp_path / "gpx",
        today=date(2026, 6, 1),
    )
    assert enriched.elev_gain_ft is None
    assert enriched.elev_source is None
    assert any("Elevation unknown" in warning for warning in enriched.warnings)
