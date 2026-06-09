"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from api.routes.races import router as races_router

settings = get_settings()

app = FastAPI(title="Race Predictor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(races_router)
app.include_router(predict_router)
