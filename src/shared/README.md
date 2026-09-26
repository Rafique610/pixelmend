# src/shared/

Shared infrastructure reused across all four tasks.

| File | Purpose |
|------|---------|
| `config.py` | Pydantic Settings singleton — `get_settings()` |
| `tracking.py` | MLflow logging helpers |
| `optuna_utils.py` | Optuna study factory + MLflow bridge |
| `losses.py` | L1, SSIM, combined reconstruction loss |
| `metrics.py` | PSNR, SSIM evaluation metrics |
| `visualization.py` | Reconstruction grids, error maps, training curves |
| `corruptions.py` | Runtime corruption pipeline (clean/SP/blur/occlusion) |
| `manifests.py` | Deterministic val/test manifest generator and loader |
| `datasets/pets.py` | Oxford-IIIT Pet dataset (Tasks 1–3) |
| `datasets/fs2k.py` | FS2K face-to-sketch dataset (Task 4) |
| `datasets/corrupted.py` | Corruption-wrapping dataset adapter |
