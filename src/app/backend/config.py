"""Application configuration settings for PixelMend FastAPI backend."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings with GENAI_ environment variable prefix."""

    app_name: str = "PixelMend Generative Vision API"
    version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Model file paths (relative to repo root or absolute in container)
    models_dir: str = "models/onnx"
    task1_model_name: str = "task1_universal_ae.onnx"
    task2_classifier_name: str = "task2_classifier.onnx"
    task2_blur_name: str = "task2_specialist_blur.onnx"
    task2_occlusion_name: str = "task2_specialist_occlusion.onnx"
    task2_salt_name: str = "task2_specialist_salt.onnx"
    task3_moe_name: str = "task3_soft_moe.onnx"
    task4_generator_name: str = "task4_generator.onnx"

    prefer_cuda: bool = True

    model_config = SettingsConfigDict(
        env_prefix="GENAI_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()
