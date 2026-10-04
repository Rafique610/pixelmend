"""Soft Mixture-of-Experts (MoE) Restoration API router for Task 3 workspace."""

import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.app.backend.config import get_settings
from src.app.backend.schemas.task3 import SoftMoERestoreResponse
from src.app.backend.services.corruption import apply_synthetic_corruption
from src.app.backend.services.inference import session_manager
from src.app.backend.services.preprocessing import (
    ImageValidationError,
    compute_error_map_base64,
    pil_to_base64_data_url,
    postprocess_nchw_to_pil,
    preprocess_image_to_nchw,
    validate_image_bytes,
)

router = APIRouter(prefix="/api/v1/restore", tags=["Soft MoE Restoration"])

EXPERT_KEYS = [
    "identity_clean",
    "expert_salt",
    "expert_blur",
    "expert_occlusion",
]


@router.post(
    "/soft-moe",
    response_model=SoftMoERestoreResponse,
    summary="Soft Mixture-of-Experts continuous restoration",
    description=(
        "Executes single-graph differentiable Soft MoE restoration. A continuous gating network "
        "computes temperature-scaled softmax weights over all 4 expert branches "
        "(identity bypass, salt-and-pepper, blur, occlusion), synthesizing a weighted output."
    ),
)
async def restore_soft_moe(
    file: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, WEBP)"),
    apply_corruption: bool = Form(
        default=False,
        description="Whether to synthetically corrupt a clean input on the server",
    ),
    corruption_type: Optional[str] = Form(
        default=None,
        description="Corruption type: 'clean', 'salt_and_pepper', 'gaussian_blur', 'occlusion'",
    ),
    severity: Optional[float] = Form(
        default=1.0,
        description="Corruption severity level (1=mild, 2=medium, 3=severe)",
    ),
) -> SoftMoERestoreResponse:
    """Execute single-graph Soft MoE inference pipeline."""
    settings = get_settings()

    if not session_manager.has_model("task3_soft_moe"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task 3 Soft MoE ONNX model is not loaded in memory.",
        )

    try:
        raw_bytes = await file.read()
        pil_original = validate_image_bytes(
            raw_bytes, max_size_bytes=settings.max_upload_size_bytes
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image payload: {exc}",
        ) from exc

    # Preprocess image into float32 NCHW [1, 3, 128, 128]
    orig_nchw = preprocess_image_to_nchw(pil_original)
    original_data_url = pil_to_base64_data_url(pil_original)

    # Apply synthetic degradation if requested
    corruption_meta = None
    if apply_corruption:
        c_type = corruption_type or "gaussian_blur"
        try:
            input_nchw, corruption_meta = apply_synthetic_corruption(
                orig_nchw,
                corruption_type=c_type,
                severity=severity if severity is not None else 1.0,
            )
            corrupted_pil = postprocess_nchw_to_pil(input_nchw)
            corrupted_data_url = pil_to_base64_data_url(corrupted_pil)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    else:
        input_nchw = orig_nchw
        corrupted_data_url = original_data_url

    # Execute single-graph Soft MoE ONNX model
    t_start = time.perf_counter_ns()
    outputs = session_manager.run_inference(
        "task3_soft_moe",
        {"input_image": input_nchw},
    )
    t_end = time.perf_counter_ns()
    inference_time_ms = round((t_end - t_start) / 1e6, 2)

    restored_nchw = outputs[0]  # shape [1, 3, 128, 128]
    weights_vector = outputs[1][0]  # shape [4]

    # Post-process routing weights
    routing_weights = {
        EXPERT_KEYS[i]: round(float(weights_vector[i]), 4)
        for i in range(len(EXPERT_KEYS))
    }
    dominant_idx = int(np.argmax(weights_vector))
    dominant_expert = EXPERT_KEYS[dominant_idx]
    dominant_weight = round(float(weights_vector[dominant_idx]), 4)

    # Calculate routing entropy: H = -sum(w * log2(w + eps))
    eps = 1e-9
    entropy = round(float(-np.sum(weights_vector * np.log2(weights_vector + eps))), 4)

    # Convert restored tensor to PIL and base64
    restored_pil = postprocess_nchw_to_pil(restored_nchw)
    restored_data_url = pil_to_base64_data_url(restored_pil)

    # Generate residual difference error heatmaps
    error_map = compute_error_map_base64(restored_nchw, input_nchw, colormap_name="turbo")
    clean_error_map = (
        compute_error_map_base64(restored_nchw, orig_nchw, colormap_name="turbo")
        if apply_corruption
        else None
    )

    return SoftMoERestoreResponse(
        original_image=original_data_url,
        corrupted_image=corrupted_data_url,
        restored_image=restored_data_url,
        error_map=error_map,
        clean_error_map=clean_error_map,
        routing_weights=routing_weights,
        dominant_expert=dominant_expert,
        dominant_weight=dominant_weight,
        entropy=entropy,
        inference_time_ms=inference_time_ms,
        corruption_applied=corruption_meta,
    )
