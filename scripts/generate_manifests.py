"""
scripts/generate_manifests.py
-----------------------------
Generates reproducible deterministic JSON manifests:
1. manifests/val_manifest.json (736 images, 25% balanced across 4 corruption classes)
2. manifests/test_manifest.json (3,669 images x 10 standardized evaluations = 36,690 instances)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings
from src.shared.manifests import build_test_manifest, build_val_manifest, save_manifest


def main():
    settings = get_settings()
    manifests_dir = Path(settings.manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = manifests_dir / "pets_split.json"

    if not split_manifest.is_file():
        print(f"Error: Base split manifest not found at {split_manifest}")
        print("Please run scripts/download_pets.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Generating Deterministic Validation & Test Manifests")
    print("=" * 60)

    # 1. Validation Manifest
    print("\n1. Building validation manifest (736 images, 4 corruption types balanced)...")
    t0 = time.time()
    val_manifest = build_val_manifest(split_manifest, seed=42, image_size=128)
    val_out = manifests_dir / "val_manifest.json"
    save_manifest(val_manifest, val_out)
    print(f"   Saved {len(val_manifest['entries'])} entries to {val_out} ({time.time() - t0:.2f}s)")
    dist = val_manifest["metadata"]["class_distribution"]
    print(f"   Distribution: {dist}")

    # 2. Test Benchmark Manifest
    print("\n2. Building test benchmark manifest (3,669 images x 10 = 36,690 entries)...")
    t0 = time.time()
    test_manifest = build_test_manifest(split_manifest, seed=42, image_size=128)
    test_out = manifests_dir / "test_manifest.json"
    save_manifest(test_manifest, test_out, compact=True)
    num_evals = test_manifest["metadata"]["total_evaluation_instances"]
    print(f"   Saved {num_evals} evaluation instances to {test_out} ({time.time() - t0:.2f}s)")

    print("\nManifest generation completed successfully!")


if __name__ == "__main__":
    main()
