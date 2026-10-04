"""Backend services for inference, preprocessing, and model management."""

from src.app.backend.services.inference import ModelSessionManager, session_manager
from src.app.backend.services.preprocessing import (
    ImageValidationError,
    base64_to_pil,
    compute_error_map_base64,
    pil_to_base64_data_url,
    postprocess_nchw_to_pil,
    preprocess_image_to_nchw,
    validate_image_bytes,
)

__all__ = [
    "ModelSessionManager",
    "session_manager",
    "ImageValidationError",
    "validate_image_bytes",
    "preprocess_image_to_nchw",
    "postprocess_nchw_to_pil",
    "pil_to_base64_data_url",
    "base64_to_pil",
    "compute_error_map_base64",
]
