"""Race-day temperature from Open-Meteo (US °F)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from race_predictor.constants import DEFAULT_TEMP_F

FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"

US_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
FORECAST_HORIZON_DAYS = 16
CLIMATOLOGY_YEARS = 10
REQUEST_TIMEOUT_SEC = 10

JsonFetcher = Callable[[str], dict]


@dataclass(frozen=True)
class RaceDayWeather:
    temp_f: float
    source: str  # forecast | archive | typical | model_default


def geocode_us_city(
    city: str | None,
    state: str | None,
    *,
    fetch_json: JsonFetcher | None = None,
) -> tuple[float, float] | None:
    """Resolve a US city/state to latitude/longitude via Open-Meteo geocoding."""
    if not city:
        return None

    loader = fetch_json or _fetch_json
    params = {
        "name": city.strip(),
        "count": 10,
        "language": "en",
        "format": "json",
        "countryCode": "US",
    }
    url = f"{GEOCODING_BASE_URL}?{urlencode(params)}"
    try:
        payload = loader(url)
    except (URLError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        return None

    state_name = US_STATE_NAMES.get((state or "").strip().upper(), (state or "").strip())
    if state_name:
        for hit in results:
            if not isinstance(hit, dict):
                continue
            admin1 = str(hit.get("admin1") or "").strip()
            if admin1.casefold() == state_name.casefold():
                lat = hit.get("latitude")
                lon = hit.get("longitude")
                if lat is not None and lon is not None:
                    return float(lat), float(lon)

    first = results[0]
    if not isinstance(first, dict):
        return None
    lat = first.get("latitude")
    lon = first.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def fetch_race_day_weather(
    latitude: float,
    longitude: float,
    race_date: date,
    *,
    default_temp_f: float = DEFAULT_TEMP_F,
    today: date | None = None,
    fetch_json: JsonFetcher | None = None,
) -> RaceDayWeather:
    """Estimate race-day temperature in °F for a lat/lon and date."""
    reference = today or date.today()
    loader = fetch_json or _fetch_json

    if race_date < reference:
        temp = _daily_mean_from_api(
            loader,
            ARCHIVE_BASE_URL,
            latitude,
            longitude,
            race_date,
            race_date,
        )
        if temp is not None:
            return RaceDayWeather(temp_f=temp, source="archive")
    elif race_date <= reference + timedelta(days=FORECAST_HORIZON_DAYS):
        temp = _daily_mean_from_api(
            loader,
            FORECAST_BASE_URL,
            latitude,
            longitude,
            race_date,
            race_date,
        )
        if temp is not None:
            return RaceDayWeather(temp_f=temp, source="forecast")
    else:
        temp = _climatology_temp_f(loader, latitude, longitude, race_date, reference)
        if temp is not None:
            return RaceDayWeather(temp_f=temp, source="typical")

    return RaceDayWeather(temp_f=default_temp_f, source="model_default")


def _climatology_temp_f(
    loader: JsonFetcher,
    latitude: float,
    longitude: float,
    race_date: date,
    today: date,
) -> float | None:
    samples: list[float] = []
    for year in range(today.year - CLIMATOLOGY_YEARS, today.year):
        try:
            sample_date = _same_calendar_day(year, race_date.month, race_date.day)
        except ValueError:
            continue
        temp = _daily_mean_from_api(
            loader,
            ARCHIVE_BASE_URL,
            latitude,
            longitude,
            sample_date,
            sample_date,
        )
        if temp is not None:
            samples.append(temp)
    if not samples:
        return None
    return sum(samples) / len(samples)


def _same_calendar_day(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError:
        # e.g. Feb 29 → Feb 28 on non-leap years
        if month == 2 and day == 29:
            return date(year, 2, 28)
        raise


def _daily_mean_from_api(
    loader: JsonFetcher,
    base_url: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> float | None:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    }
    url = f"{base_url}?{urlencode(params)}"
    try:
        payload = loader(url)
    except (URLError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return _parse_daily_mean(payload)


def _parse_daily_mean(payload: dict) -> float | None:
    daily = payload.get("daily") or {}
    max_temps = daily.get("temperature_2m_max") or []
    min_temps = daily.get("temperature_2m_min") or []
    if not max_temps or not min_temps:
        return None
    max_f = max_temps[0]
    min_f = min_temps[0]
    if max_f is None or min_f is None:
        return None
    return (float(max_f) + float(min_f)) / 2.0


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.load(response)
