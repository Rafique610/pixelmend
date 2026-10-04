"""Image preprocessing, validation, normalization, and base64 encoding services."""

import base64
import io
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ImageValidationError(ValueError):
    """Raised when uploaded image fails format, dimension, or size checks."""

    status_code: int = 400


class ImageTooLargeError(ImageValidationError):
    """Raised when the upload exceeds the maximum allowed size (HTTP 413)."""

    status_code = 413


def validate_image_bytes(
    data: bytes,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> Image.Image:
    """Validate image bytes against size limits and supported formats, returning RGB PIL Image."""
    if len(data) == 0:
        raise ImageValidationError("Uploaded image file is empty (0 bytes).")
    if len(data) > max_size_bytes:
        size_mb = len(data) / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ImageTooLargeError(
            f"File size exceeds limit: {size_mb:.2f} MB exceeds maximum {max_mb:.1f} MB."
        )

    try:
        bio = io.BytesIO(data)
        img = Image.open(bio)
        img.verify()
        bio.seek(0)
        img = Image.open(bio)
    except Exception as exc:
        raise ImageValidationError(f"Invalid or corrupted image format: {exc}") from exc

    if img.format not in ("JPEG", "PNG", "WEBP"):
        raise ImageValidationError(
            f"Unsupported image format '{img.format}'. Supported formats: JPEG, PNG, WEBP."
        )

    return img.convert("RGB")


def preprocess_image_to_nchw(
    image: Image.Image,
    target_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Resize image to target_size (128x128) and return float32 NCHW tensor normalized to [0, 1]."""
    resized = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    # arr is [H, W, C] -> transpose to [C, H, W] -> add batch dim [1, C, H, W]
    nchw = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(nchw, dtype=np.float32)


def postprocess_nchw_to_pil(tensor: np.ndarray) -> Image.Image:
    """Convert float32 NCHW (or CHW) tensor in [0, 1] to RGB PIL Image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = np.clip(tensor, 0.0, 1.0)
    hwc = np.transpose(tensor, (1, 2, 0))
    uint8_arr = (hwc * 255.0).round().astype(np.uint8)
    return Image.fromarray(uint8_arr, mode="RGB")


def pil_to_base64_data_url(image: Image.Image, img_format: str = "PNG") -> str:
    """Encode PIL Image into a base64 Data URL string."""
    buf = io.BytesIO()
    image.save(buf, format=img_format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if img_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def base64_to_pil(base64_str: str) -> Image.Image:
    """Decode a base64 Data URL or raw base64 string into RGB PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    data = base64.b64decode(base64_str)
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img.convert("RGB")


def compute_error_map_base64(
    original_nchw: np.ndarray,
    restored_nchw: np.ndarray,
    colormap_name: str = "turbo",
) -> str:
    """Compute absolute residual difference heatmap and return as base64 PNG data URL."""
    orig = original_nchw[0] if original_nchw.ndim == 4 else original_nchw
    rest = restored_nchw[0] if restored_nchw.ndim == 4 else restored_nchw
    diff = np.abs(rest - orig)
    # Average across RGB channels -> [H, W] in [0, 1]
    diff_2d = np.mean(diff, axis=0)
    diff_clipped = np.clip(diff_2d, 0.0, 1.0)

    # Apply matplotlib colormap (turbo or jet)
    cmap = plt.get_cmap(colormap_name)
    colored = cmap(diff_clipped)  # RGBA in [0, 1]
    rgb = (colored[..., :3] * 255.0).astype(np.uint8)

    heatmap_pil = Image.fromarray(rgb, mode="RGB")
    return pil_to_base64_data_url(heatmap_pil, img_format="PNG")
