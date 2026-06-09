"""Race prediction route (manual conditions or predict-by-race_id)."""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException

from api.config import Settings, get_settings
from api.schemas import PredictRequest, PredictResponse, PredictionItem
from race_predictor.data.loader import load_runs
from race_predictor.data.race_enrichment import enrich_race
from race_predictor.data.runsignup_client import RunSignupClient, RunSignupError
from race_predictor.models.predictor import load_model, predict_all, predict_race

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_races(
    body: PredictRequest,
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    csv_path = settings.data_dir / "activities.csv"
    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No activities.csv found in {settings.data_dir}/",
        )

    runs = load_runs(csv_path)
    if not runs:
        raise HTTPException(status_code=400, detail="No runs found in activities.csv")

    if not settings.model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Run `race-predictor train --population` first.",
        )
    model = load_model(settings.model_path)

    race_id = body.race_id
    race_name: str | None = None
    elev_source: str | None = None
    warnings: list[str] = []
    elev_gain_ft = body.elev_gain_ft
    elev_loss_ft = body.elev_loss_ft
    temp_f = body.temp_f
    temp_source = "request" if body.temp_f is not None else "model_default"
    as_of_date = body.as_of
    distance_labels: list[str] | None = None

    if race_id is not None:
        client = RunSignupClient()
        try:
            enriched = enrich_race(
                client,
                race_id,
                as_of=body.as_of,
                overrides_path=settings.overrides_path,
                cache_dir=settings.enrichment_cache_dir,
                gpx_dir=settings.gpx_dir,
            )
        except RunSignupError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        race_name = enriched.name
        elev_source = enriched.elev_source
        warnings = list(enriched.warnings)
        if elev_gain_ft is None:
            elev_gain_ft = enriched.elev_gain_ft
        if elev_loss_ft is None:
            elev_loss_ft = enriched.elev_loss_ft
        if body.temp_f is None and enriched.temp_f is not None:
            temp_f = enriched.temp_f
            temp_source = enriched.weather_source or "enrichment"
        if as_of_date is None:
            as_of_date = enriched.race_date

        if body.event_id is not None:
            matching = [
                event
                for event in enriched.offered_events
                if event.event_id == body.event_id
            ]
            if not matching:
                raise HTTPException(
                    status_code=400,
                    detail=f"event_id {body.event_id} not found for race {race_id}",
                )
            distance_labels = [matching[0].distance_label]

    if as_of_date is not None and as_of_date < date.today():
        raise HTTPException(
            status_code=400,
            detail=f"as_of must be today ({date.today():%Y-%m-%d}) or a future date",
        )

    if elev_gain_ft is None or elev_loss_ft is None:
        raise HTTPException(
            status_code=400,
            detail="Elevation is unknown for this race. Provide elev_gain_ft and elev_loss_ft.",
        )

    as_of_dt = (
        datetime.combine(as_of_date, datetime.min.time())
        if as_of_date is not None
        else runs[-1].date
    )
    if temp_f is None:
        temp_f = model.default_temp_f
        if body.temp_f is None and race_id is None:
            temp_source = "model_default"

    if distance_labels:
        predictions = []
        for label in distance_labels:
            prediction = predict_race(
                runs,
                model,
                as_of_dt,
                label,
                elev_gain_ft=elev_gain_ft,
                elev_loss_ft=elev_loss_ft,
                temp_f=temp_f,
            )
            if prediction is not None:
                predictions.append(prediction)
    else:
        predictions = predict_all(
            runs,
            model,
            as_of_dt,
            elev_gain_ft=elev_gain_ft,
            elev_loss_ft=elev_loss_ft,
            temp_f=temp_f,
        )

    if not predictions:
        raise HTTPException(
            status_code=400,
            detail="Could not produce predictions — insufficient training data in the lookback window",
        )

    return PredictResponse(
        as_of=as_of_dt.date(),
        elev_gain_ft=elev_gain_ft,
        elev_loss_ft=elev_loss_ft,
        temp_f=temp_f,
        temp_source=temp_source,
        race_id=race_id,
        race_name=race_name,
        elev_source=elev_source,
        warnings=warnings,
        predictions=[
            PredictionItem(
                distance_label=item.distance_label,
                predicted_time_sec=item.predicted_time_sec,
                pace_min_per_mi=item.pace_min_per_mi,
                interval_low_sec=item.interval_low_sec,
                interval_high_sec=item.interval_high_sec,
                confidence=item.confidence,
            )
            for item in predictions
        ],
    )
