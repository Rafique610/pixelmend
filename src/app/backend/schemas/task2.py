"""Request and response schemas for Task 2: Hard-Routed Restoration workspace."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LatencyBreakdown(BaseModel):
    """Execution latency decomposition across stages in milliseconds."""

    classifier_ms: float = Field(description="Stage 1 CNN classification latency (ms)")
    specialist_ms: float = Field(description="Stage 2 Specialist autoencoder latency (ms)")
    total_ms: float = Field(description="End-to-end total pipeline latency (ms)")


class HardRoutedRestoreResponse(BaseModel):
    """Response schema for Task 2 hard-routed sequential restoration endpoint."""

    original_image: str = Field(description="Base64 Data URL of uploaded original image")
    corrupted_image: str = Field(description="Base64 Data URL of input image fed to classifier")
    restored_image: str = Field(description="Base64 Data URL of reconstructed restored output")
    error_map: str = Field(
        description="Base64 Data URL of difference heatmap (|restored - corrupted|)"
    )
    class_probabilities: Dict[str, float] = Field(
        description="Softmax class probability distribution across all 4 corruption classes"
    )
    predicted_class: str = Field(description="Highest-probability corruption class name")
    confidence: float = Field(description="Maximum softmax probability score (0.0 to 1.0)")
    selected_expert: str = Field(
        description="Routed specialist autoencoder filename or 'Identity Bypass'"
    )
    inference_time: LatencyBreakdown = Field(description="Structured latency decomposition")
    corruption_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Synthetic corruption metadata if server-generated, or null if direct upload",
    )
