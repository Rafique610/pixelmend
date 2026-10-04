"""Request and response schemas for Task 3: Soft Mixture-of-Experts (MoE) workspace."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SoftMoERestoreResponse(BaseModel):
    """Response schema for Task 3 Soft MoE differentiable restoration endpoint."""

    original_image: str = Field(description="Base64 Data URL of uploaded original image")
    corrupted_image: str = Field(description="Base64 Data URL of input image fed to MoE network")
    restored_image: str = Field(description="Base64 Data URL of reconstructed restored output")
    error_map: str = Field(
        description="Base64 Data URL of difference heatmap (|restored - corrupted|)"
    )
    clean_error_map: Optional[str] = Field(
        default=None,
        description="Base64 Data URL of residual heatmap against clean reference",
    )
    routing_weights: Dict[str, float] = Field(
        description="Continuous gating weights across all 4 experts summing to 1.0"
    )
    dominant_expert: str = Field(
        description="Expert identifier receiving the highest gating weight allocation"
    )
    dominant_weight: float = Field(
        description="Gating weight of the dominant expert (0.0 to 1.0)"
    )
    entropy: float = Field(
        description="Shannon entropy of the routing weight distribution in bits"
    )
    inference_time_ms: float = Field(
        description="Single-graph ONNX execution latency in milliseconds"
    )
    corruption_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Synthetic corruption metadata if server-generated, or null if direct upload",
    )
