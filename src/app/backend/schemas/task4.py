"""Request and response schemas for Task 4: Face-to-Sketch synthesis workspace."""

from pydantic import BaseModel, Field


class SketchGenerateResponse(BaseModel):
    """Response schema for Task 4 style-conditioned face-to-sketch synthesis."""

    original_image: str = Field(description="Base64 Data URL of uploaded input face photo")
    sketch_image: str = Field(description="Base64 Data URL of synthesized facial sketch")
    selected_style: int = Field(description="FS2K target style index (1, 2, or 3)")
    style_description: str = Field(description="Human-readable description of the sketch style")
    inference_time_ms: float = Field(description="Generator ONNX inference latency in milliseconds")
