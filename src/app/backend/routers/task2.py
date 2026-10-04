"""Hard-Routed Restoration API router for Task 2 deep learning workspace."""

import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.app.backend.config import get_settings
from src.app.backend.schemas.task2 import HardRoutedRestoreResponse, LatencyBreakdown
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

router = APIRouter(prefix="/api/v1/restore", tags=["Hard-Routed Restoration"])

CLASS_NAMES = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
EXPERT_MODELS = [
    "Identity Bypass",
    "task2_specialist_salt",
    "task2_specialist_blur",
    "task2_specialist_occlusion",
]


@router.post(
    "/hard-routed",
    response_model=HardRoutedRestoreResponse,
    summary="Two-stage classified routing to specialist autoencoder",
    description=(
        "Executes sequential hard routing: Stage 1 convolutional classifier predicts "
        "corruption class probabilities; Stage 2 routes exclusively to the corresponding "
        "specialist autoencoder (or executes zero-latency identity bypass for clean images)."
    ),
)
async def restore_hard_routed(
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
    force_expert: Optional[str] = Form(
        default=None,
        description="Optional manual expert override: 'clean', 'salt', 'blur', 'occlusion'",
    ),
) -> HardRoutedRestoreResponse:
    """Execute two-stage hard-routed restoration pipeline."""
    settings = get_settings()

    # Verify classifier and specialists are loaded
    if not session_manager.has_model("task2_classifier"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task 2 Classifier ONNX model is not loaded in memory.",
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

    # Stage 1: Classification & Softmax Probabilities
    t_cls_start = time.perf_counter_ns()
    cls_outputs = session_manager.run_inference("task2_classifier", {"input": input_nchw})
    t_cls_end = time.perf_counter_ns()
    classifier_ms = round((t_cls_end - t_cls_start) / 1e6, 2)

    logits = cls_outputs[0]  # shape [1, 4]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    prob_vector = probs[0]

    class_probabilities = {CLASS_NAMES[i]: round(float(prob_vector[i]), 4) for i in range(4)}
    pred_idx = int(np.argmax(prob_vector))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = round(float(prob_vector[pred_idx]), 4)

    # Resolve active specialist (auto routing or manual override)
    routed_idx = pred_idx
    if force_expert:
        forced = force_expert.lower().strip()
        if "clean" in forced:
            routed_idx = 0
        elif "salt" in forced:
            routed_idx = 1
        elif "blur" in forced:
            routed_idx = 2
        elif "occl" in forced:
            routed_idx = 3

    # Stage 2: Specialist Routing Execution
    t_spec_start = time.perf_counter_ns()
    if routed_idx == 0:
        # Identity bypass: no specialist invocation needed
        restored_nchw = input_nchw.copy()
        selected_expert_name = "Identity Bypass"
        specialist_ms = 0.0
    else:
        expert_key = EXPERT_MODELS[routed_idx]
        if not session_manager.has_model(expert_key):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Routed specialist model '{expert_key}' is not loaded.",
            )
        spec_outputs = session_manager.run_inference(expert_key, {"input": input_nchw})
        restored_nchw = spec_outputs[0]
        selected_expert_name = f"{expert_key}.onnx"
        t_spec_end = time.perf_counter_ns()
        specialist_ms = round((t_spec_end - t_spec_start) / 1e6, 2)

    total_ms = round(classifier_ms + specialist_ms, 2)

    # Convert restored tensor to PIL and Base64 Data URL
    restored_pil = postprocess_nchw_to_pil(restored_nchw)
    restored_data_url = pil_to_base64_data_url(restored_pil)

    # Generate residual absolute difference heatmap
    error_map_url = compute_error_map_base64(input_nchw, restored_nchw, colormap_name="turbo")

    return HardRoutedRestoreResponse(
        original_image=original_data_url,
        corrupted_image=corrupted_data_url,
        restored_image=restored_data_url,
        error_map=error_map_url,
        class_probabilities=class_probabilities,
        predicted_class=predicted_class,
        confidence=confidence,
        selected_expert=selected_expert_name,
        inference_time=LatencyBreakdown(
            classifier_ms=classifier_ms,
            specialist_ms=specialist_ms,
            total_ms=total_ms,
        ),
        corruption_applied=corruption_meta,
    )
