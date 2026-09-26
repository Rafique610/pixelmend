# GenAI Assignment — Domain Index

## Project Overview

| Key | Value |
|-----|-------|
| Course | AI-4009 Generative AI |
| Assignment | #1 — Autoencoders, MoE, Conditional GAN |
| Framework | PyTorch (≥ 2.1) |
| HPO | Optuna (SQLite-backed) |
| Tracking | MLflow (local, self-hosted) |
| Export | ONNX (opset ≥ 17) + ONNX Runtime inference |
| Backend | FastAPI + ONNX Runtime |
| Frontend | React + Tailwind CSS (Vite) |
| UI Design | Google Stitch (required evidence in report) |
| Deployment | Docker Compose (frontend + backend containers) |
| Report | IEEE format, LaTeX |

## Datasets

### Tasks 1–3: Oxford-IIIT Pet Dataset
- 37 categories, cats and dogs.
- Use official trainval as development data.
- Split: 80% train / 20% val, `random_seed=42`.
- Official test set: untouched until final evaluation.
- Images: RGB, resized to 128×128.
- Same split for all three tasks.

### Task 4: FS2K Facial Sketch Synthesis Dataset
- 2,104 paired face photos ↔ sketches, 3 style categories.
- Use official train/test split.
- Reserve 15% of training as validation, `random_seed=42`, stratified by style.
- Images: 128×128, preserve photo-sketch pairing.

## Corruption Pipeline (Tasks 1–3)

| Corruption | Training Params | Test Severities |
|------------|----------------|-----------------|
| Clean | Identity (no corruption) | — |
| Salt-and-pepper | p ∈ U(0.02, 0.15) | 0.03, 0.08, 0.15 |
| Gaussian blur | kernel ∈ {3,5,7}, σ ∈ U(0.5, 2.5) | (3,0.7), (5,1.5), (7,2.5) |
| Rectangular occlusion | 1–3 masks, 10–35% area | ~10%, ~20%, ~35% (1,2,3 rects) |

- Training: dynamic (random per load).
- Validation/Test: deterministic manifests (JSON), generated once.

## Architecture Decisions (Locked)

These come directly from the assignment spec and are non-negotiable:

- Task 1: Conv encoder → bottleneck → conv decoder. Combined L1 + SSIM loss.
- Task 2: Conv classifier (cross-entropy) → 3 specialist autoencoders.
  Oracle-routing and predicted-routing evaluation modes.
- Task 3: Softmax gating over identity + 3 experts. Initialize from Task 2.
  Warm-up → joint fine-tune. Loss: λ₁L1 + λ₂(1−SSIM) + λ₃CE + λ₄balance.
- Task 4: U-Net generator + PatchGAN discriminator. Style embedding for 3
  categories. Loss: adversarial + λ_L1 · L1.

## Architecture Decisions (Open — Require Research)

- Task 1: Bottleneck dimension, skip connections (if/how), channel progression.
- Task 1: α (L1 vs SSIM weighting) — Optuna.
- Task 2: Classifier architecture (ResNet-like? simple conv stack?).
- Task 3: Temperature τ, balance regularizer form, fine-tune LR schedule.
- Task 4: Style embedding dimension, how to inject style (concat? FiLM? AdaIN?).
- Task 4: λ_L1 value — Optuna.
- Cross-cutting: Optuna pruner choice.

## Application Workspaces

| # | Workspace Name | Backend Endpoint | Features |
|---|---------------|-----------------|----------|
| 1 | Universal Restoration | `/api/v1/restore/universal` | Upload/select + apply corruption + restore + error map |
| 2 | Hard-Routed Restoration | `/api/v1/restore/hard-routed` | Classifier probs, predicted class, selected expert, result |
| 3 | Soft Mixture-of-Experts Restoration | `/api/v1/restore/soft-moe` | 4 routing weights, expert contribution visual, result |
| 4 | Face-to-Sketch Generator | `/api/v1/sketch/generate` | Photo upload/webcam, style select (1/2/3), side-by-side |

## Phase Status

| Phase | Status |
|-------|--------|
| Bootstrapping (docs + invariants) | ✅ Done |
| Plans (all 6 files written) | ✅ Done |
| Setup (shared infra) | ⬜ Not started — `docs/plans/setup.md` ready |
| Task 1 | ⬜ Not started — `docs/plans/task1-universal-autoencoder.md` ready |
| Task 2 | ⬜ Not started — `docs/plans/task2-hard-routing.md` ready |
| Task 3 | ⬜ Not started — `docs/plans/task3-soft-moe.md` ready |
| Task 4 | ⬜ Not started — `docs/plans/task4-face-to-sketch.md` ready |
| App (React + FastAPI + Docker) | ⬜ Not started — `docs/plans/app.md` ready |
| IEEE Report | ⬜ Not started |
| Demo Video | ⬜ Not started |
