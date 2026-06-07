"""API configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_path: Path
    cors_origins: list[str]


DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def get_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        model_path=Path(os.getenv("MODEL_PATH", "models/trained_model.pkl")),
        cors_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    )
