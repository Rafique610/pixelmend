"""
src/shared/config.py
--------------------
Single source of truth for all runtime configuration.
All values come from environment variables (GENAI_ prefix) or .env file.
Never read os.environ directly anywhere else in the project.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="GENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = Field(
        default="",
        description="Force device: 'cuda', 'cpu', 'mps', or '' for auto",
    )

    # ------------------------------------------------------------------
    # Paths (relative to project root)
    # ------------------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    checkpoints_dir: Path = Field(default=Path("checkpoints"))
    onnx_dir: Path = Field(default=Path("models/onnx"))
    manifests_dir: Path = Field(default=Path("manifests"))
    optuna_db: Path = Field(default=Path("optuna/optuna_studies.db"))
    results_dir: Path = Field(default=Path("results"))

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="",
        description="MLflow tracking URI. Empty = local mlruns/ directory.",
    )

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173")

    # ------------------------------------------------------------------
    # Training defaults
    # ------------------------------------------------------------------
    num_workers: int = Field(default=2)
    seed: int = Field(default=42)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cuda", "cpu", "mps", ""}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def resolved_device(self) -> str:
        """Return the actual torch device string (auto-detect if not forced)."""
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.resolved_device)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def optuna_db_url(self) -> str:
        """SQLite URL for Optuna storage."""
        return f"sqlite:///{self.optuna_db.resolve()}"

    @property
    def mlflow_uri(self) -> str:
        """Resolved MLflow tracking URI (falls back to local mlruns/)."""
        return self.mlflow_tracking_uri or ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton. Import this everywhere."""
    return Settings()
