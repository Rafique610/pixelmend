"""Request and response schemas for Task 1: Universal Restoration workspace."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UniversalRestoreResponse(BaseModel):
    """Response schema for Task 1 universal image restoration endpoint."""

    original_image: str = Field(description="Base64 Data URL of uploaded original image")
    corrupted_image: str = Field(
        description="Base64 Data URL of corrupted image fed to autoencoder"
    )
    restored_image: str = Field(description="Base64 Data URL of reconstructed restored output")
    error_map: str = Field(
        description="Base64 Data URL of residual difference heatmap (|restored - corrupted|)"
    )
    clean_error_map: Optional[str] = Field(
        default=None,
        description="Base64 Data URL of residual difference heatmap (|restored - clean|)",
    )
    corruption_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata on synthetic degradation applied, or null if direct corrupted upload",
    )
    inference_time_ms: float = Field(description="Model execution latency in milliseconds")
    model_name: str = Field(
        default="task1_universal_ae.onnx", description="Active ONNX model identifier"
    )
