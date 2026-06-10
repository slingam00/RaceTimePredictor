"""RunSignup race search and enrichment routes."""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.config import Settings, get_settings
from api.horizon import load_prediction_horizon
from api.schemas import (
    PredictionHorizonResponse,
    RaceDetail,
    RaceEventDetail,
    RaceSearchResponse,
    RaceSummary,
)
from race_predictor.data.race_enrichment import EnrichedRace, enrich_race, offered_distance_labels
from race_predictor.data.runsignup_client import RunSignupClient, RunSignupError, RunSignupRaceSummary

router = APIRouter(prefix="/api", tags=["races"])


@router.get("/prediction-horizon", response_model=PredictionHorizonResponse)
def get_prediction_horizon(
    settings: Settings = Depends(get_settings),
) -> PredictionHorizonResponse:
    horizon = load_prediction_horizon(settings)
    if horizon.max_prediction_date is None:
        raise HTTPException(
            status_code=404,
            detail=f"No activities.csv found in {settings.data_dir}/",
        )
    return PredictionHorizonResponse(
        max_prediction_date=horizon.max_prediction_date,
        prediction_horizon_message=horizon.message,
    )


@router.get("/races/search", response_model=RaceSearchResponse)
def search_races(
    q: Optional[str] = Query(default=None, description="Race name search"),
    city: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    page: int = Query(default=1, ge=1),
    results_per_page: int = Query(default=25, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> RaceSearchResponse:
    reference = date.today()
    effective_start = start_date if start_date is not None else reference
    if effective_start < reference:
        effective_start = reference

    horizon = load_prediction_horizon(settings)
    max_date = horizon.max_prediction_date

    if max_date is not None and effective_start > max_date:
        return RaceSearchResponse(
            races=[],
            page=page,
            results_per_page=results_per_page,
            max_prediction_date=max_date,
            prediction_horizon_message=horizon.message,
        )

    client = RunSignupClient()
    try:
        result = client.search_races(
            name=q,
            city=city,
            state=state,
            start_date=effective_start,
            end_date=end_date,
            max_date=max_date,
            page=page,
            results_per_page=results_per_page,
            today=reference,
        )
    except RunSignupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RaceSearchResponse(
        races=[_summary_from_search(race) for race in result.races],
        page=result.page,
        results_per_page=result.results_per_page,
        max_prediction_date=max_date,
        prediction_horizon_message=horizon.message,
    )


@router.get("/races/{race_id}", response_model=RaceDetail)
def get_race_detail(
    race_id: int,
    settings: Settings = Depends(get_settings),
) -> RaceDetail:
    client = RunSignupClient()
    try:
        enriched = enrich_race(
            client,
            race_id,
            overrides_path=settings.overrides_path,
            cache_dir=settings.enrichment_cache_dir,
            gpx_dir=settings.gpx_dir,
        )
    except RunSignupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _detail_from_enriched(enriched)


def _summary_from_search(race: RunSignupRaceSummary) -> RaceSummary:
    return RaceSummary(
        race_id=race.race_id,
        name=race.name,
        city=race.city,
        state=race.state,
        next_date=race.next_date,
        offered_distances=offered_distance_labels(race.events),
    )


def _detail_from_enriched(enriched: EnrichedRace) -> RaceDetail:
    return RaceDetail(
        race_id=enriched.race_id,
        name=enriched.name,
        city=enriched.city,
        state=enriched.state,
        race_date=enriched.race_date,
        elev_gain_ft=enriched.elev_gain_ft,
        elev_loss_ft=enriched.elev_loss_ft,
        elev_source=enriched.elev_source,
        temp_f=enriched.temp_f,
        weather_source=enriched.weather_source,
        offered_events=[
            RaceEventDetail(
                event_id=event.event_id,
                name=event.name,
                distance_label=event.distance_label,
                distance_mi=event.distance_mi,
            )
            for event in enriched.offered_events
        ],
        warnings=enriched.warnings,
    )
