"""Universal Restoration API router for Task 1 deep learning workspace."""

import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.app.backend.config import get_settings
from src.app.backend.schemas.task1 import UniversalRestoreResponse
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

router = APIRouter(prefix="/api/v1/restore", tags=["Universal Restoration"])


@router.post(
    "/universal",
    response_model=UniversalRestoreResponse,
    summary="Restore corrupted image using Universal Autoencoder",
    description=(
        "Processes an uploaded image through the universal convolutional autoencoder. "
        "Supports direct upload of pre-corrupted photos, or on-the-fly synthetic corruption "
        "(salt-and-pepper noise, Gaussian blur, rectangular occlusion) for clean images."
    ),
)
async def restore_universal(
    file: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, WEBP)"),
    apply_corruption: bool = Form(
        default=False,
        description="Whether to synthetically corrupt a clean input on the server",
    ),
    corruption_type: Optional[str] = Form(
        default=None,
        description="Corruption type: 'salt_and_pepper', 'gaussian_blur', 'rectangular_occlusion'",
    ),
    severity: Optional[float] = Form(
        default=1.0,
        description="Corruption severity level (1=mild, 2=medium, 3=severe, or custom parameter)",
    ),
) -> UniversalRestoreResponse:
    """Execute Universal Autoencoder restoration pipeline."""
    settings = get_settings()

    if not session_manager.has_model("task1_universal_ae"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task 1 Universal Autoencoder ONNX model is not loaded in memory.",
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

    # Preprocess original image into float32 NCHW [1, 3, 128, 128]
    orig_nchw = preprocess_image_to_nchw(pil_original)
    original_data_url = pil_to_base64_data_url(pil_original)

    # Apply synthetic degradation if requested
    corruption_meta = None
    clean_error_map_url = None

    if apply_corruption:
        c_type = corruption_type or "gaussian_blur"
        try:
            corrupted_nchw, corruption_meta = apply_synthetic_corruption(
                orig_nchw,
                corruption_type=c_type,
                severity=severity if severity is not None else 1.0,
            )
            corrupted_pil = postprocess_nchw_to_pil(corrupted_nchw)
            corrupted_data_url = pil_to_base64_data_url(corrupted_pil)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    else:
        corrupted_nchw = orig_nchw
        corrupted_data_url = original_data_url

    # Execute ONNX Runtime inference
    t0 = time.perf_counter_ns()
    outputs = session_manager.run_inference("task1_universal_ae", {"input": corrupted_nchw})
    t_end = time.perf_counter_ns()
    inference_time_ms = round((t_end - t0) / 1e6, 2)

    restored_nchw = outputs[0]
    restored_pil = postprocess_nchw_to_pil(restored_nchw)
    restored_data_url = pil_to_base64_data_url(restored_pil)

    # Compute residual absolute difference error maps
    error_map_url = compute_error_map_base64(corrupted_nchw, restored_nchw, colormap_name="turbo")
    if apply_corruption:
        clean_error_map_url = compute_error_map_base64(
            orig_nchw, restored_nchw, colormap_name="turbo"
        )

    return UniversalRestoreResponse(
        original_image=original_data_url,
        corrupted_image=corrupted_data_url,
        restored_image=restored_data_url,
        error_map=error_map_url,
        clean_error_map=clean_error_map_url,
        corruption_applied=corruption_meta,
        inference_time_ms=inference_time_ms,
        model_name="task1_universal_ae.onnx",
    )
