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


def get_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        model_path=Path(os.getenv("MODEL_PATH", "models/trained_model.pkl")),
        cors_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    )
