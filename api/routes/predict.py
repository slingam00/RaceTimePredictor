"""Race prediction route (manual elevation and temperature)."""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException

from api.config import Settings, get_settings
from api.schemas import PredictRequest, PredictResponse, PredictionItem
from race_predictor.data.loader import load_runs
from race_predictor.models.predictor import load_model, predict_all, train

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

    if body.as_of is not None and body.as_of < date.today():
        raise HTTPException(
            status_code=400,
            detail=f"as_of must be today ({date.today():%Y-%m-%d}) or a future date",
        )

    if settings.model_path.exists():
        model = load_model(settings.model_path)
    else:
        model = train(runs, settings.model_path)

    as_of_dt = datetime.combine(body.as_of, datetime.min.time()) if body.as_of else runs[-1].date
    temp_f = body.temp_f if body.temp_f is not None else model.default_temp_f
    temp_source = "request" if body.temp_f is not None else "model_default"

    predictions = predict_all(
        runs,
        model,
        as_of_dt,
        elev_gain_ft=body.elev_gain_ft,
        elev_loss_ft=body.elev_loss_ft,
        temp_f=temp_f,
    )
    if not predictions:
        raise HTTPException(
            status_code=400,
            detail="Could not produce predictions — insufficient training data in the lookback window",
        )

    return PredictResponse(
        as_of=as_of_dt.date(),
        elev_gain_ft=body.elev_gain_ft,
        elev_loss_ft=body.elev_loss_ft,
        temp_f=temp_f,
        temp_source=temp_source,
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
