"""Face-to-Sketch Synthesis API router for Task 4 conditional GAN workspace."""

import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image

from src.app.backend.config import get_settings
from src.app.backend.schemas.task4 import SketchGenerateResponse
from src.app.backend.services.inference import session_manager
from src.app.backend.services.preprocessing import (
    ImageValidationError,
    pil_to_base64_data_url,
    preprocess_image_to_nchw,
    validate_image_bytes,
)

router = APIRouter(prefix="/api/v1/sketch", tags=["Face-to-Sketch Synthesis"])

STYLE_DESCRIPTIONS = {
    0: "FS2K Style 1 - Clean Line / Pencil Contour",
    1: "FS2K Style 2 - Dense Shading / Cross-Hatch Sketch",
    2: "FS2K Style 3 - Fine Tonal / Shaded Art Sketch",
}


@router.post(
    "/generate",
    response_model=SketchGenerateResponse,
    summary="Synthesize style-conditioned facial sketch from photo",
    description=(
        "Executes conditional GAN generator inference on uploaded face photo. "
        "Conditions synthesis on chosen FS2K style (1=clean contour, 2=cross-hatching, "
        "3=tonal shading) using embedded FiLM modulation at the U-Net bottleneck."
    ),
)
async def generate_sketch(
    file: UploadFile = File(..., description="Uploaded face photo (JPEG, PNG, WEBP)"),
    style: int = Form(
        default=1,
        description="Target sketch style (1=Pencil Line, 2=Cross-Hatch, 3=Tonal Shading)",
    ),
) -> SketchGenerateResponse:
    """Execute Task 4 style-conditioned face-to-sketch inference."""
    settings = get_settings()

    # Map user style (accepting both 1-indexed [1, 2, 3] and 0-indexed [0, 1, 2])
    if style in (1, 2, 3):
        style_idx = style - 1
        display_style = style
    elif style in (0, 1, 2):
        style_idx = style
        display_style = style + 1
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid style selection: {style}. Expected 1, 2, or 3.",
        )

    if not session_manager.has_model("task4_generator"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task 4 Generator ONNX model is not loaded in memory.",
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

    # Preprocess face photo: [1, 3, 128, 128] normalized to [-1, 1]
    nchw_01 = preprocess_image_to_nchw(pil_original)
    photo_tensor = (nchw_01 * 2.0 - 1.0).astype(np.float32)
    style_tensor = np.array([style_idx], dtype=np.int64)

    original_data_url = pil_to_base64_data_url(pil_original)

    # Execute generator ONNX session
    t_start = time.perf_counter_ns()
    outputs = session_manager.run_inference(
        "task4_generator",
        {"photo": photo_tensor, "style_index": style_tensor},
    )
    t_end = time.perf_counter_ns()
    inference_time_ms = round((t_end - t_start) / 1e6, 2)

    # Postprocess sketch tensor from [-1, 1] back to uint8 [0, 255] RGB
    sketch_tensor = outputs[0]  # shape [1, 3, 128, 128]
    sketch_01 = np.clip((sketch_tensor[0] + 1.0) / 2.0, 0.0, 1.0)
    sketch_rgb = (sketch_01.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    sketch_pil = Image.fromarray(sketch_rgb, mode="RGB")
    sketch_data_url = pil_to_base64_data_url(sketch_pil, img_format="PNG")

    return SketchGenerateResponse(
        original_image=original_data_url,
        sketch_image=sketch_data_url,
        selected_style=display_style,
        style_description=STYLE_DESCRIPTIONS[style_idx],
        inference_time_ms=inference_time_ms,
    )
