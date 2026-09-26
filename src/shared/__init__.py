# src/shared — Shared infrastructure reused across all four tasks.
#
# Modules:
#   config.py          — Pydantic Settings singleton (get_settings())
#   tracking.py        — MLflow logging wrappers
#   optuna_utils.py    — Optuna study factory + trial-to-MLflow bridge
#   losses.py          — L1, SSIM, combined reconstruction loss
#   metrics.py         — PSNR, SSIM evaluation metrics
#   visualization.py   — Reconstruction grids, error maps, training curves
#   corruptions.py     — Runtime corruption pipeline (all 4 types)
#   manifests.py       — Deterministic val/test manifest generator & loader
#   datasets/          — PetDataset, FS2KDataset, CorruptedDatasetWrapper
