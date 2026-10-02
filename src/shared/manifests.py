"""
src/shared/manifests.py
-----------------------
Generation, validation, and loading of deterministic manifests for validation
and test benchmarks on the Oxford-IIIT Pet dataset.

Manifests:
- manifests/val_manifest.json:
    736 validation images, strictly balanced (25% Clean, 25% S&P, 25% Blur, 25% Occlusion).
    Pre-sampled deterministic parameters (seed 42) for reproducible validation metrics.

- manifests/test_manifest.json:
    3,669 test images x 10 standardized evaluations = 36,690 total evaluation instances.
    Each test image evaluated under:
      - 1 Clean baseline
      - 3 Salt-and-Pepper severities (p in {0.03, 0.08, 0.15})
      - 3 Gaussian Blur severities ((k, sigma) in {(3, 0.7), (5, 1.5), (7, 2.5)})
      - 3 Rectangular Occlusion severities (1 box ~10%, 2 boxes ~20%, 3 boxes ~35%)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional, Union

from src.shared.config import Settings, get_settings
from src.shared.corruptions import (
    TEST_SEVERITIES,
    CorruptionLabel,
    CorruptionType,
    sample_occlusion_boxes,
)


def build_val_manifest(
    split_manifest_path: Union[str, Path],
    seed: int = 42,
    image_size: int = 128,
) -> dict[str, Any]:
    """
    Construct a deterministic, class-balanced validation manifest from the split manifest.
    Guarantees exactly 25% distribution across the 4 corruption types (184 images each).
    """
    split_path = Path(split_manifest_path)
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    val_items = data.get("val") or data.get("splits", {}).get("val", [])
    num_val = len(val_items)
    if num_val != 736:
        raise ValueError(f"Expected 736 validation images, got {num_val}")

    rng = random.Random(seed)

    # 4 corruption types, 184 samples each
    types_list = [
        CorruptionType.CLEAN,
        CorruptionType.SALT_AND_PEPPER,
        CorruptionType.GAUSSIAN_BLUR,
        CorruptionType.OCCLUSION,
    ]
    per_type = num_val // len(types_list)
    corruption_assignments = []
    for c_t in types_list:
        corruption_assignments.extend([c_t] * per_type)

    # Shuffle the assignments deterministically so images don't cluster by corruption
    rng.shuffle(corruption_assignments)

    entries = []
    for idx, (item, c_type) in enumerate(zip(val_items, corruption_assignments)):
        label = CorruptionLabel[c_type.name].value
        img_file = item.get("image") or item.get("image_path", "")

        if c_type == CorruptionType.CLEAN:
            params = {}

        elif c_type == CorruptionType.SALT_AND_PEPPER:
            p = round(rng.uniform(0.02, 0.15), 4)
            params = {"p": p, "channel_independent": False}

        elif c_type == CorruptionType.GAUSSIAN_BLUR:
            k = rng.choice([3, 5, 7])
            sigma = round(rng.uniform(0.5, 2.5), 3)
            params = {"kernel_size": k, "sigma": sigma}

        elif c_type == CorruptionType.OCCLUSION:
            num_boxes = rng.choice([1, 2, 3])
            target_ratio = round(rng.uniform(0.10, 0.35), 3)
            boxes = sample_occlusion_boxes(
                height=image_size,
                width=image_size,
                num_boxes=num_boxes,
                target_ratio=target_ratio,
                rng=rng,
            )
            params = {"boxes": boxes, "fill_value": 0.0, "target_ratio": target_ratio}

        entries.append(
            {
                "val_id": idx,
                "image": img_file,
                "class_id": item.get("class_id", 0),
                "species": item.get("species", 0),
                "breed_id": item.get("breed_id", 0),
                "corruption_type": c_type.value,
                "corruption_label": label,
                "params": params,
            }
        )

    return {
        "metadata": {
            "split": "val",
            "seed": seed,
            "total_samples": num_val,
            "class_distribution": {c.value: per_type for c in types_list},
        },
        "entries": entries,
    }


def build_test_manifest(
    split_manifest_path: Union[str, Path],
    seed: int = 42,
    image_size: int = 128,
) -> dict[str, Any]:
    """
    Construct the full 10-variation benchmark test manifest for all 3,669 test images.
    Total evaluation instances: 3,669 * 10 = 36,690.
    """
    split_path = Path(split_manifest_path)
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_items = data.get("test") or data.get("splits", {}).get("test", [])
    num_test = len(test_items)
    if num_test != 3669:
        raise ValueError(f"Expected 3669 test images, got {num_test}")

    rng = random.Random(seed)
    entries = []
    eval_id = 0

    for img_idx, item in enumerate(test_items):
        img_file = item.get("image") or item.get("image_path", "")
        class_id = item.get("class_id", 0)
        species = item.get("species", 0)
        breed_id = item.get("breed_id", 0)

        # 1. Clean (variation 0)
        entries.append(
            {
                "eval_id": eval_id,
                "image_id": img_idx,
                "image": img_file,
                "class_id": class_id,
                "species": species,
                "breed_id": breed_id,
                "corruption_type": CorruptionType.CLEAN.value,
                "corruption_label": CorruptionLabel.CLEAN.value,
                "severity": 0,
                "variant_name": "clean",
                "params": {},
            }
        )
        eval_id += 1

        # 2. Salt-and-Pepper (variations 1, 2, 3)
        for sp_spec in TEST_SEVERITIES[CorruptionType.SALT_AND_PEPPER.value]:
            entries.append(
                {
                    "eval_id": eval_id,
                    "image_id": img_idx,
                    "image": img_file,
                    "class_id": class_id,
                    "species": species,
                    "breed_id": breed_id,
                    "corruption_type": CorruptionType.SALT_AND_PEPPER.value,
                    "corruption_label": CorruptionLabel.SALT_AND_PEPPER.value,
                    "severity": sp_spec["severity"],
                    "variant_name": sp_spec["name"],
                    "params": {"p": sp_spec["p"], "channel_independent": False},
                }
            )
            eval_id += 1

        # 3. Gaussian Blur (variations 4, 5, 6)
        for blur_spec in TEST_SEVERITIES[CorruptionType.GAUSSIAN_BLUR.value]:
            entries.append(
                {
                    "eval_id": eval_id,
                    "image_id": img_idx,
                    "image": img_file,
                    "class_id": class_id,
                    "species": species,
                    "breed_id": breed_id,
                    "corruption_type": CorruptionType.GAUSSIAN_BLUR.value,
                    "corruption_label": CorruptionLabel.GAUSSIAN_BLUR.value,
                    "severity": blur_spec["severity"],
                    "variant_name": blur_spec["name"],
                    "params": {
                        "kernel_size": blur_spec["kernel_size"],
                        "sigma": blur_spec["sigma"],
                    },
                }
            )
            eval_id += 1

        # 4. Rectangular Occlusion (variations 7, 8, 9)
        for occl_spec in TEST_SEVERITIES[CorruptionType.OCCLUSION.value]:
            boxes = sample_occlusion_boxes(
                height=image_size,
                width=image_size,
                num_boxes=occl_spec["num_boxes"],
                target_ratio=occl_spec["target_ratio"],
                rng=rng,
            )
            entries.append(
                {
                    "eval_id": eval_id,
                    "image_id": img_idx,
                    "image": img_file,
                    "class_id": class_id,
                    "species": species,
                    "breed_id": breed_id,
                    "corruption_type": CorruptionType.OCCLUSION.value,
                    "corruption_label": CorruptionLabel.OCCLUSION.value,
                    "severity": occl_spec["severity"],
                    "variant_name": occl_spec["name"],
                    "params": {
                        "boxes": boxes,
                        "fill_value": 0.0,
                        "target_ratio": occl_spec["target_ratio"],
                    },
                }
            )
            eval_id += 1

    return {
        "metadata": {
            "split": "test",
            "seed": seed,
            "num_test_images": num_test,
            "variations_per_image": 10,
            "total_evaluation_instances": eval_id,
            "severities_definition": TEST_SEVERITIES,
        },
        "entries": entries,
    }


def save_manifest(
    manifest: dict[str, Any],
    output_path: Union[str, Path],
    compact: bool = False,
) -> Path:
    """Save manifest to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        if compact:
            json.dump(manifest, f, separators=(",", ":"))
        else:
            json.dump(manifest, f, indent=2)
    return out


def load_val_manifest(
    path: Optional[Union[str, Path]] = None,
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    """Load the validation manifest entries."""
    cfg = settings or get_settings()
    p = Path(path or cfg.manifests_dir / "val_manifest.json")
    if not p.is_file():
        raise FileNotFoundError(
            f"Validation manifest not found at {p}. Run scripts/generate_manifests.py first."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def load_test_manifest(
    path: Optional[Union[str, Path]] = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Load the full test manifest structure (metadata + entries)."""
    cfg = settings or get_settings()
    p = Path(path or cfg.manifests_dir / "test_manifest.json")
    if not p.is_file():
        raise FileNotFoundError(
            f"Test manifest not found at {p}. Run scripts/generate_manifests.py first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
