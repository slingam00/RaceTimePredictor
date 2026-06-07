"""Tests for GPX elevation profile parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from race_predictor.data.gpx_profile import parse_gpx, parse_gpx_text

FIXTURE = Path("tests/fixtures/simple_hill.gpx")


def test_parse_gpx_simple_hill():
    profile = parse_gpx(FIXTURE)
    # 0 -> 30.48 m (100 ft) up, then 30.48 -> 15.24 m (50 ft) down
    assert profile.elev_gain_ft == pytest.approx(100.0, rel=1e-3)
    assert profile.elev_loss_ft == pytest.approx(50.0, rel=1e-3)
    assert profile.point_count == 3
    assert profile.source == "gpx"


def test_parse_gpx_text_matches_file():
    xml = FIXTURE.read_text(encoding="utf-8")
    from_file = parse_gpx(FIXTURE)
    from_text = parse_gpx_text(xml)
    assert from_text == from_file


def test_parse_gpx_rejects_missing_elevation():
    gpx_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <gpx version="1.1"><trk><trkseg>
      <trkpt lat="0" lon="0"/>
    </trkseg></trk></gpx>"""
    with pytest.raises(ValueError, match="at least two points"):
        parse_gpx_text(gpx_xml)


def test_noise_threshold_ignores_small_bumps():
    gpx_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <gpx version="1.1"><trk><trkseg>
      <trkpt lat="0" lon="0"><ele>0</ele></trkpt>
      <trkpt lat="0" lon="0"><ele>0.9144</ele></trkpt>
      <trkpt lat="0" lon="0"><ele>0</ele></trkpt>
    </trkseg></trk></gpx>"""
    # 0.9144 m ≈ 3 ft — at default threshold should be ignored both ways
    profile = parse_gpx_text(gpx_xml, noise_threshold_ft=3.0)
    assert profile.elev_gain_ft == pytest.approx(0.0, abs=0.01)
    assert profile.elev_loss_ft == pytest.approx(0.0, abs=0.01)
