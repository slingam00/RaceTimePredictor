"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
def root() -> dict:
    return {
        "service": "Race Predictor API",
        "ui": "http://localhost:3000",
        "endpoints": ["/health", "POST /api/predict"],
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
