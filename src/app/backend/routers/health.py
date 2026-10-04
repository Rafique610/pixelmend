"""Health check and model status router for PixelMend backend."""

from datetime import datetime, timezone

from fastapi import APIRouter

from src.app.backend.config import get_settings
from src.app.backend.schemas.common import HealthResponse
from src.app.backend.services.inference import session_manager

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return backend operational health, ONNX execution provider, and loaded model status."""
    settings = get_settings()
    status, active_provider, models_loaded = session_manager.get_health_status()
    now_utc = datetime.now(timezone.utc).isoformat()

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.version,
        execution_provider=active_provider,
        models_loaded=models_loaded,
        timestamp_utc=now_utc,
    )
