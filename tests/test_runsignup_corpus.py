"""Tests for RunSignup corpus collection."""

from __future__ import annotations

import json
from pathlib import Path

from race_predictor.data.runsignup_client import (
    RunSignupClient,
    RunSignupCredentials,
    RunSignupEvent,
)
from race_predictor.data.runsignup_corpus import (
    build_coverage_report,
    load_catalog,
    results_to_runs,
    select_event,
    sync_corpus,
    write_corpus_csv,
)
from race_predictor.data.weather import RaceDayWeather


FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class MockRunSignupClient(RunSignupClient):
    def __init__(self) -> None:
        super().__init__(
            credentials=RunSignupCredentials(api_key="test", api_secret="secret"),
            fetch_json=self._fetch,
        )

    def _fetch(self, url: str, headers: dict | None = None) -> dict:
        if "/race/74589/results/get-results" in url:
            return _load_fixture("runsignup_results_74589_101.json")
        if "/race/74589" in url:
            return _load_fixture("runsignup_race_74589.json")
        raise AssertionError(f"Unexpected URL: {url}")


def test_load_catalog():
    catalog = load_catalog(FIXTURES / "catalog_test.json")
    assert len(catalog) == 1
    assert catalog[0].distance_label == "5K"
    assert catalog[0].course_type == "flat"


def test_select_event_matches_distance_label():
    events = [
        RunSignupEvent(101, "Cambridge 5K", 3.1, "M", "10:00:00"),
        RunSignupEvent(102, "Cambridge Half Marathon", 13.1, "M", "07:00:00"),
    ]
    catalog = load_catalog(FIXTURES / "catalog_test.json")[0]
    selected = select_event(events, catalog)
    assert selected is not None
    assert selected.event_id == 101


def test_results_to_runs_maps_finishers():
    from race_predictor.data.runsignup_client import RunSignupResult

    results = [
        RunSignupResult(9001, 101, 1, 1110.0, "Alex", "Runner"),
        RunSignupResult(9002, 102, 2, 1155.0, "Blair", "Jogger"),
    ]
    runs, skipped = results_to_runs(
        results,
        race_id=74589,
        event_id=101,
        race_name="Cambridge",
        event_name="Cambridge 5K",
        race_date=__import__("datetime").datetime(2024, 11, 3),
        distance_mi=3.10686,
        elev_gain_ft=50,
        elev_loss_ft=50,
        temp_f=55.0,
        course_type="flat",
    )
    assert len(runs) == 2
    assert skipped == 0
    assert runs[0].athlete_id == "9001"
    assert runs[0].is_likely_race is True
    assert runs[0].elev_gain_ft == 50


def test_sync_corpus_writes_csv_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "race_predictor.data.runsignup_corpus.fetch_race_day_weather",
        lambda *args, **kwargs: RaceDayWeather(temp_f=58.0, source="archive"),
    )

    output = tmp_path / "corpus.csv"
    report_path = tmp_path / "coverage.json"
    runs, report = sync_corpus(
        MockRunSignupClient(),
        catalog_path=FIXTURES / "catalog_test.json",
        output_path=output,
        coverage_report_path=report_path,
    )

    assert len(runs) == 3
    assert output.is_file()
    assert report_path.is_file()
    assert report.by_distance["5K"]["finishers"] == 3
    assert report.by_course_type["flat"] == 3

    write_corpus_csv(runs, tmp_path / "rewrite.csv")
    text = (tmp_path / "rewrite.csv").read_text(encoding="utf-8")
    assert "athlete_id" in text
    assert "9001" in text


def test_coverage_report_flags_gaps():
    from race_predictor.data.runsignup_corpus import RaceSyncStats

    report = build_coverage_report(
        runs=[],
        race_stats=[
            RaceSyncStats(
                catalog_entry="Test 5K",
                race_id=1,
                distance_label="5K",
                course_type="flat",
                finishers=10,
            )
        ],
    )
    assert not report.passed
    assert any("5K" in gap for gap in report.gaps)
