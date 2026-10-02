"""
scripts/download_fs2k.py
------------------------
Automated downloader and preprocessor for the FS2K (Facial Sketch Synthesis 2K)
paired face-to-sketch dataset.

1. Downloads the official FS2K dataset (109.7 MB zip archive) from Google Drive:
   File ID: 1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp
2. Unpacks into data/fs2k/
3. Validates 1:1 photo-sketch pairing across all 2,104 pairs (0 missing):
   - Style 0: 976 pairs
   - Style 1: 731 pairs
   - Style 2: 397 pairs
4. Constructs deterministic stratified 85% train / 15% val split (seed 42),
   preserving official test set (1,046 pairs) strictly intact.
5. Saves manifests/fs2k_split.json.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.shared.config import get_settings

GDRIVE_FILE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"


def download_gdrive_file(file_id: str, dest_path: Path) -> Path:
    """Download large Google Drive file with virus-scan confirmation bypass."""
    print(f"Connecting to Google Drive for file ID: {file_id}...")
    session = requests.Session()
    url = f"https://drive.google.com/uc?id={file_id}"
    r = session.get(url)

    if "drive.usercontent.google.com/download" in r.text or "download-form" in r.text:
        match_action = re.search(r'action="([^"]+)"', r.text)
        match_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)
        match_confirm = re.search(r'name="confirm"\s+value="([^"]+)"', r.text)

        action = (
            match_action.group(1)
            if match_action
            else "https://drive.usercontent.google.com/download"
        )
        uuid_val = match_uuid.group(1) if match_uuid else ""
        confirm_val = match_confirm.group(1) if match_confirm else "t"

        params = {"id": file_id, "confirm": confirm_val}
        if uuid_val:
            params["uuid"] = uuid_val

        download_resp = session.get(action, params=params, stream=True)
    else:
        download_resp = session.get(
            f"https://drive.google.com/uc?export=download&id={file_id}",
            stream=True,
        )

    download_resp.raise_for_status()

    total_size = int(download_resp.headers.get("content-length", 0))
    print(f"Downloading archive ({total_size / (1024 * 1024):.1f} MB)...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    t0 = time.time()

    with open(dest_path, "wb") as f:
        for chunk in download_resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total_size * 100) if total_size else 0
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {mb:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\nDownload complete in {elapsed:.1f}s ({downloaded / (1024 * 1024):.1f} MB).")
    return dest_path


def extract_fs2k(zip_path: Path, extract_dir: Path) -> Path:
    """Extract FS2K zip archive into target directory."""
    print(f"Extracting {zip_path.name} to {extract_dir}...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extraction complete in {time.time() - t0:.1f}s.")

    # Check if extracted inside nested FS2K/ folder
    nested = extract_dir / "FS2K"
    if nested.is_dir() and (nested / "photo").is_dir():
        for item in nested.iterdir():
            target = extract_dir / item.name
            if not target.exists():
                item.rename(target)

    return extract_dir


def resolve_pair(fs2k_dir: Path, item: dict) -> dict:
    """Resolve exact file paths for photo and sketch from annotation record."""
    p_rel = item["image_name"]
    folder, img_stem = p_rel.split("/")
    num_str = img_stem.replace("image", "")
    s_rel = folder.replace("photo", "sketch") + "/sketch" + num_str

    # Resolve photo extension (.jpg or .png)
    p_file = fs2k_dir / "photo" / (p_rel + ".jpg")
    if not p_file.exists():
        p_file = fs2k_dir / "photo" / (p_rel + ".png")

    # Resolve sketch extension (.jpg or .png)
    s_file = fs2k_dir / "sketch" / (s_rel + ".jpg")
    if not s_file.exists():
        s_file = fs2k_dir / "sketch" / (s_rel + ".png")

    if not p_file.exists():
        raise FileNotFoundError(f"Missing photo file for {p_rel}")
    if not s_file.exists():
        raise FileNotFoundError(f"Missing sketch file for {s_rel}")

    entry = dict(item)
    entry["photo_path"] = str(p_file.relative_to(fs2k_dir)).replace("\\", "/")
    entry["sketch_path"] = str(s_file.relative_to(fs2k_dir)).replace("\\", "/")
    entry["style_id"] = int(item.get("style", 0))
    entry["stem"] = img_stem
    return entry


def build_fs2k_splits(fs2k_dir: Path, val_ratio: float = 0.15, seed: int = 42) -> dict:
    """Build stratified 85/15 train/val split from anno_train.json; preserve anno_test.json."""
    anno_train_file = fs2k_dir / "anno_train.json"
    anno_test_file = fs2k_dir / "anno_test.json"

    if not anno_train_file.is_file() or not anno_test_file.is_file():
        raise FileNotFoundError(f"Official annotations not found in {fs2k_dir}")

    with open(anno_train_file, "r", encoding="utf-8") as f:
        raw_train = json.load(f)
    with open(anno_test_file, "r", encoding="utf-8") as f:
        raw_test = json.load(f)

    print(f"Loaded {len(raw_train)} raw train entries and {len(raw_test)} raw test entries.")

    resolved_trainval = [resolve_pair(fs2k_dir, it) for it in raw_train]
    resolved_test = [resolve_pair(fs2k_dir, it) for it in raw_test]

    # Stratify official trainval into Train and Val (15% validation)
    rng = random.Random(seed)
    by_style: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for item in resolved_trainval:
        by_style[item["style_id"]].append(item)

    final_train: list[dict] = []
    final_val: list[dict] = []

    print("\nStratified partition of official training set (15% validation):")
    for s_id in sorted(by_style.keys()):
        items = by_style[s_id]
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val_sub = items[:n_val]
        train_sub = items[n_val:]

        final_val.extend(val_sub)
        final_train.extend(train_sub)
        pct = len(val_sub) / len(items) * 100
        print(
            f"  Style {s_id} (Style {s_id + 1}): "
            f"{len(train_sub)} train, {len(val_sub)} val ({pct:.1f}%)"
        )

    # Style counts in test
    test_by_style: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for item in resolved_test:
        test_by_style[item["style_id"]] += 1
    print(f"Official test partition style counts: {test_by_style}")

    manifest = {
        "metadata": {
            "dataset": "FS2K",
            "seed": seed,
            "val_ratio": val_ratio,
            "train_count": len(final_train),
            "val_count": len(final_val),
            "test_count": len(resolved_test),
            "total_pairs": len(final_train) + len(final_val) + len(resolved_test),
            "style_labels": {
                0: "Style 1 (photo1)",
                1: "Style 2 (photo2)",
                2: "Style 3 (photo3)",
            },
        },
        "train": final_train,
        "val": final_val,
        "test": resolved_test,
    }
    return manifest


def main():
    settings = get_settings()
    fs2k_dir = Path(settings.data_dir) / "fs2k"
    fs2k_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = Path(settings.manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FS2K Paired Face-to-Sketch Dataset Setup")
    print("=" * 60)

    # 1. Download if photo/ does not exist
    photo_dir = fs2k_dir / "photo"
    sketch_dir = fs2k_dir / "sketch"

    if not (photo_dir.is_dir() and sketch_dir.is_dir()):
        zip_path = fs2k_dir / "FS2K.zip"
        if not zip_path.is_file():
            download_gdrive_file(GDRIVE_FILE_ID, zip_path)
        extract_fs2k(zip_path, fs2k_dir)
        if zip_path.is_file():
            zip_path.unlink()
    else:
        print(f"FS2K dataset already present at {fs2k_dir}.")

    # 2. Build and save splits
    manifest = build_fs2k_splits(fs2k_dir, val_ratio=0.15, seed=42)
    manifest_path = manifests_dir / "fs2k_split.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    meta = manifest["metadata"]
    print(f"\nManifest successfully saved to {manifest_path}")
    print(f"  - Train pairs: {meta['train_count']}")
    print(f"  - Val pairs:   {meta['val_count']}")
    print(f"  - Test pairs:  {meta['test_count']}")
    print(f"  - Total pairs: {meta['total_pairs']}")
    print("FS2K dataset setup complete!")


if __name__ == "__main__":
    main()
