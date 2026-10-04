"""Common Pydantic schemas for request validation, responses, and errors."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(description="System status ('ok' or 'degraded')")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Application version")
    execution_provider: str = Field(description="Active ONNX Runtime execution provider")
    models_loaded: Dict[str, bool] = Field(description="Status of all 7 expected ONNX models")
    timestamp_utc: str = Field(description="UTC timestamp of the health check")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    error: str = Field(description="Brief error category or message")
    detail: Optional[str] = Field(default=None, description="Detailed actionable diagnostics")
    code: int = Field(default=400, description="HTTP status code")


class RootInfoResponse(BaseModel):
    """API root informational schema."""

    app_name: str
    version: str
    docs_url: str
    health_url: str
    status: str
