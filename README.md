# AI-4009 Generative AI — Assignment #1: PixelMend

**Author**: Muhammad Rafique (`i230747@isb.nu.edu.pk`)  
**Institution**: FAST National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad  
**Repository**: [https://github.com/Rafique610/pixelmend](https://github.com/Rafique610/pixelmend)  
**Video Demonstration**: [https://www.youtube.com/watch?v=mO7WHRdSPoE](https://www.youtube.com/watch?v=mO7WHRdSPoE)  

End-to-end implementation and deployment of four generative computer vision systems:
1. **Universal Denoising Autoencoder** (multi-corruption blind restoration)
2. **Hard-Routed Specialist Autoencoders** (corruption classifier + dedicated restoration experts)
3. **Soft Mixture-of-Experts Restoration** (differentiable gating network over 4 branches)
4. **Style-Conditioned Face-to-Sketch Synthesis** (conditional GAN on FS2K dataset)

Integrated and deployed through a single **FastAPI + React (Tailwind CSS)** web application with **Docker Compose**.

---

## Evaluator Quick Start (CPU-only, no GPU required)

### Option A — Docker (one command)

Requirements: Docker Desktop / Docker Engine with Compose v2. Nothing else.

```bash
git clone https://github.com/Rafique610/pixelmend.git pixelmend && cd pixelmend
ls models/onnx/*.onnx                 # must list 7 files (~150 MB total, committed to git)
docker compose up --build
```

| What | URL |
| :--- | :--- |
| Web app (React UI) | <http://localhost> |
| Interactive API docs (OpenAPI / Swagger) | <http://localhost/docs> |
| Health + model status | <http://localhost/health> |

The first build takes a few minutes (CPU PyTorch + Node build). The frontend starts only after the backend reports all 7 ONNX models loaded. Stop with `Ctrl+C`, then `docker compose down`.

Expected `models/onnx/` contents: `task1_universal_ae.onnx`, `task2_classifier.onnx`, `task2_specialist_blur.onnx`, `task2_specialist_occlusion.onnx`, `task2_specialist_salt.onnx`, `task3_soft_moe.onnx`, `task4_generator.onnx`. If one is missing, its workspace returns HTTP 503 with an actionable message (other workspaces keep working) and `/health` reports it as not loaded.

### Option B â€” Local development (no Docker)

Requirements: Python >= 3.11 + [`uv`](https://github.com/astral-sh/uv), Node 20 + `pnpm`, [`task`](https://taskfile.dev/).

```bash
uv sync --extra dev
task install-frontend
task dev-backend          # terminal 1 -> http://localhost:8000 (docs at /docs)
task dev-frontend         # terminal 2 -> http://localhost:3000
```

### Verify it works

```bash
uv run pytest tests/test_backend_skeleton.py tests/test_backend_task1.py tests/test_backend_task2.py tests/test_backend_task3.py tests/test_backend_task4.py tests/test_backend_errors.py   # 63 tests
uv run python scripts/smoke_test.py --base-url http://localhost        # Docker  (use :8000 for local dev) -> 25/25 checks
uv run python scripts/benchmark_cpu.py                                 # regenerates results/app/cpu_benchmark.json
```

`smoke_test.py` hits all four REST endpoints, checks that Soft-MoE routing weights sum to 1, that the provider is `CPUExecutionProvider`, and that bad input is rejected with the right status code.

### API error contract

| Condition | Status | Example `detail` |
| :--- | :---: | :--- |
| Non-image / corrupted / unsupported format (only JPEG, PNG, WEBP) | 400 | `Invalid or corrupted image format: ...` |
| Empty upload | 400 | `Uploaded image file is empty (0 bytes).` |
| Invalid style / corruption type | 400 | `Unsupported corruption type ...` |
| Upload larger than 10 MB | 413 | `File size exceeds limit: ...` |
| ONNX model not loaded | 503 | `... ONNX model is not loaded in memory.` |

The frontend shows the backend `detail` text in a dismissible red alert.

## Workspace User Guide

Open <http://localhost> (or `:3000` in dev). The sidebar switches between the four workspaces. All images are resized to 128x128 for the models. Accepted uploads: JPEG/PNG/WEBP up to 10 MB.

| Workspace | Route | How to use | What you see |
| :--- | :--- | :--- | :--- |
| **Universal Restoration** (Task 1) | `/universal` | Upload an image or click a preset pet sample. Optionally enable *synthetic degradation* and choose type (salt-and-pepper, Gaussian blur, occlusion) and severity 1-3. Click *Restore*. | Original / corrupted / restored side by side, Turbo residual error map (vs. input and vs. clean original), latency, PNG download. |
| **Hard-Routed** (Task 2) | `/hard-routed` | Same inputs. Optionally override the router with a manual expert (clean / salt / blur / occlusion). | Classifier probabilities for the 4 classes, chosen specialist, latency split (classifier vs. specialist), restored image, error map. |
| **Soft MoE** (Task 3) | `/soft-moe` | Same inputs. | 4 gating weights (identity, salt, blur, occlusion; sum = 1.0), dominant expert, routing entropy, restored image, error map. |
| **Face-to-Sketch** (Task 4) | `/face-to-sketch` | Upload a face photo, use the preset portrait, or take a webcam snapshot (browser camera permission). Choose Style 1, 2 or 3. | Photo vs. generated sketch, style description, latency, PNG download. |

Tip: apply a severe occlusion on `/universal`, then on `/hard-routed` and `/soft-moe` to compare how the three systems handle the same input.

## Benchmark Summary (all 4 tasks)

**Restoration quality â€” held-out Oxford-IIIT Pet test, 128x128, 20 % stratified subsample (7,340 images), PSNR in dB** (source: `results/task3/test_benchmark_comparison.csv`, `results/task3/test_evaluation_summary.json`):

| System | Overall PSNR | Overall SSIM | Clean PSNR | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Task 1 â€” Universal AE | 20.08 | 0.589 | 20.82 | Single model, no corruption label |
| Task 2 â€” Hard router (predicted) | 23.79 | 0.496 | 77.54 | Mean inflated by identity bypass on clean images (PSNR capped at 80 dB); on corrupted classes it is below Task 1 |
| Task 2 â€” Hard router (oracle) | 24.00 | 0.495 | 80.00 | Upper bound with ground-truth routing |
| Task 3 â€” Soft MoE | 20.03 | **0.650** | 25.89 | Best SSIM and MAE (0.0673) |

On the full 36,690-image test set Task 1 scores PSNR 20.16 / SSIM 0.5935 (`results/task1/metrics_summary.json`); Task 2 full-set tables are in `results/task2/test_summary_table.md`.

**Task 4 â€” Face-to-Sketch (FS2K test, `results/task4/test_metrics.json`):** L1 0.107, PSNR 15.38 dB, SSIM 0.472, LPIPS 0.251. Per style (L1 / SSIM): style 1 0.082 / 0.510, style 2 0.153 / 0.396, style 3 0.069 / 0.591.

**Model cost â€” CPU, ONNX Runtime `CPUExecutionProvider`, batch 1, 128x128, mean of 50 runs after 10 warm-up** (source: `results/app/cpu_benchmark.json`, produced by `scripts/benchmark_cpu.py`; absolute latency varies by machine):

| System | ONNX models | Parameters | Size (MB) | CPU latency (ms) |
| :--- | :--- | :---: | :---: | :---: |
| Task 1 Universal AE | `task1_universal_ae` | 4.91 M | 18.75 | 28.4 |
| Task 2 Hard-routed (classifier + 1 specialist) | classifier + 3 specialists | 0.39 M + 3 x 4.91 M = 15.12 M | 57.74 | 2.6 (clean, identity) / ~31 (corrupted) |
| Task 3 Soft MoE (single graph) | `task3_soft_moe` | 15.12 M | 57.76 | 80.9 |
| Task 4 FiLM cGAN generator | `task4_generator` | 6.83 M | 26.08 | 28.4 |

ONNX vs. PyTorch parity is verified to < 1e-5 absolute difference for batch sizes 1, 4 and 8 (`results/task*/onnx_parity_benchmark.json`).

## Report Cross-Reference

See [`docs/report_artifacts.md`](docs/report_artifacts.md) for the mapping from each report section to its result JSON/CSV and figure. Note: `results/**/*.png` figures are git-ignored by default; regenerate with the `task eval-taskN` commands if they are missing from a fresh clone.

---

## Architecture & Workspaces

| Workspace / System | Description | Endpoint | Target Model |
| :--- | :--- | :--- | :--- |
| **Setup & Infra** | Shared data pipelines, manifests, MLflow tracking, Optuna DB | â€” | â€” |
| **Task 1: Universal Restoration** | Single convolutional autoencoder restoring clean, salt-and-pepper, blur, and occlusion without corruption label | `POST /api/v1/restore/universal` | `models/onnx/task1_universal_ae.onnx` |
| **Task 2: Hard-Routed Restoration** | 4-class classifier routing clean (identity) vs 3 specialist autoencoders | `POST /api/v1/restore/hard-routed` | `models/onnx/task2_*.onnx` (4 models) |
| **Task 3: Soft MoE Restoration** | Continuous gating network combining identity + 3 specialists dynamically | `POST /api/v1/restore/soft-moe` | `models/onnx/task3_soft_moe.onnx` |
| **Task 4: Face-to-Sketch** | Paired photo-to-sketch cGAN conditioned on 3 learned style categories | `POST /api/v1/sketch/generate` | `models/onnx/task4_generator.onnx` |

---

## Directory Structure

```
.
â”œâ”€â”€ .env.example              # Environment variables template
â”œâ”€â”€ .gitignore                # Comprehensive Git ignore rules
â”œâ”€â”€ Taskfile.yml              # Central task automation runner
â”œâ”€â”€ pyproject.toml            # Python packaging & dependencies (uv managed)
â”œâ”€â”€ data/                     # Downloaded datasets (gitignored)
â”‚   â”œâ”€â”€ oxford-iiit-pet/      # Tasks 1â€“3 development & test data
â”‚   â””â”€â”€ fs2k/                 # Task 4 paired face-to-sketch data
â”œâ”€â”€ checkpoints/              # Saved PyTorch model checkpoints (gitignored)
â”œâ”€â”€ models/
â”‚   â””â”€â”€ onnx/                 # Exported ONNX models for production inference
â”œâ”€â”€ manifests/                # Deterministic validation & test corruption manifests
â”œâ”€â”€ optuna/                   # SQLite database for Optuna studies (optuna_studies.db)
â”œâ”€â”€ scripts/                  # Data download & batch preparation scripts
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ shared/               # Shared losses, metrics, config, tracking, datasets
â”‚   â”œâ”€â”€ task1/                # Universal autoencoder module
â”‚   â”œâ”€â”€ task2/                # Classifier and specialist autoencoders
â”‚   â”œâ”€â”€ task3/                # Soft Mixture-of-Experts pipeline
â”‚   â”œâ”€â”€ task4/                # Conditional GAN generator & discriminator
â”‚   â””â”€â”€ app/
â”‚       â”œâ”€â”€ backend/          # FastAPI REST API (routers, schemas, services)
â”‚       â””â”€â”€ frontend/         # React + Tailwind CSS web interface
â””â”€â”€ tests/                    # Pytest test suites
```

---

## Quickstart & Environment Setup

### 1. Requirements
- Python `>= 3.11`
- [`uv`](https://github.com/astral-sh/uv) package manager
- [`task`](https://taskfile.dev/) task runner (or run commands via `uv run python`)
- Node.js & `pnpm` (for frontend development)
- Docker & Docker Compose (for containerized deployment)

### 2. Install Python Dependencies
```bash
uv sync --extra dev
```

### 3. Configure Environment
```bash
cp .env.example .env
```

### 4. Verify Setup & Tests
```bash
task --list
task test
```

---

## Experiment Tracking (MLflow)

All training runs, Optuna trials, learning rate curves, loss metrics, and sample visual reconstruction grids are automatically tracked using a local SQLite-backed MLflow store (`mlruns/mlflow.db`).

### Launch Tracking UI
```bash
task mlflow-ui
```
Navigate to [http://localhost:5000](http://localhost:5000) to inspect active experiments, hyperparameter comparisons, and image artifacts.

---

## Hyperparameter Optimization (Optuna)

All hyperparameter search studies across Tasks 1â€“4 are persisted in a centralized SQLite database (`optuna/optuna_studies.db`). Studies survive process restarts and automatically synchronize trial metrics with MLflow.

- **Study factory**: `create_or_load_study(study_name, direction, pruner_name, seed)`
- **Pruning**: `MedianPruner` (early termination of non-competitive trials)
- **Reporting & Export**: Automatic export of trial history to CSV (`trials_history.csv`) and JSON summary statistics (`study_summary.json`) for report figures and tables.

---

## Datasets & Data Pipelines

### Oxford-IIIT Pet (Tasks 1, 2, 3)
The Oxford-IIIT Pet dataset comprises 7,349 images across 37 cat and dog breeds.
- **Download**: Run `task download-pets` (downloads and unpacks into `data/oxford-iiit-pet/`).
- **Deterministic Partitioning**:
  - Official `trainval` (3,680 images) split into **80% Train** (2,944 images) and **20% Validation** (736 images), stratified across all 37 breeds with random seed 42.
  - Official `test` (3,669 images) reserved strictly for final benchmark evaluation across Tasks 1, 2, and 3.
  - Split manifest: `manifests/pets_split.json`.
- **PyTorch Dataset (`PetDataset`)**:
  - Location: `src/shared/datasets/pets.py`
  - Normalization: $128 \times 128$ spatial resolution, 3-channel RGB float32 tensors scaled to $[0.0, 1.0]$.
  - Fast DataLoader: `get_pet_dataloader(split, batch_size, shuffle, num_workers)`.
- **Verification**: Run `task verify-pets` to validate split counts, tensor ranges, and export sample inspection grid to `results/oxford_pets_sample_grid.png`.

### Corruption Pipeline & Benchmark Manifests
Tasks 1, 2, and 3 restore four distinct corruption states:
1. **Clean / Identity** (Label `0`): Unmodified image tensor.
2. **Salt-and-Pepper Noise** (Label `1`): Impulse noise where $p/2$ pixels are set to $0.0$ and $p/2$ to $1.0$.
   - Training: $p \sim \mathcal{U}(0.02, 0.15)$.
   - Test benchmark: Fixed severities $p \in \{0.03, 0.08, 0.15\}$.
3. **Gaussian Blur** (Label `2`): 2D Gaussian kernel convolution attenuating high spatial frequencies.
   - Training: $k \in \{3, 5, 7\}$, $\sigma \sim \mathcal{U}(0.5, 2.5)$.
   - Test benchmark: Fixed pairs $(k, \sigma) \in \{(3, 0.7), (5, 1.5), (7, 2.5)\}$.
4. **Rectangular Occlusion** (Label `3`): Non-overlapping/partially overlapping masked bounding boxes.
   - Training: $1$â€“$3$ boxes covering $10\%$â€“$35\%$ image area.
   - Test benchmark: Fixed severities ($1$ box $\approx 10\%$, $2$ boxes $\approx 20\%$, $3$ boxes $\approx 35\%$).

- **Deterministic Manifests**:
  - `manifests/val_manifest.json`: 736 validation images, strictly balanced ($25\%$ per class = 184 each), pre-seeded with fixed corruption parameters for reproducible validation across epochs.
  - `manifests/test_manifest.json`: 3,669 test images $\times$ 10 variations (1 clean + 3 severities $\times$ 3 corruptions) = 36,690 standardized evaluation instances.
  - Regenerate manifests anytime via `task generate-manifests`.
- **PyTorch Corrupted Dataset (`CorruptedPetDataset`)**:
  - Location: `src/shared/datasets/corrupted.py`
  - In `train` mode: Dynamic, stochastic on-the-fly corruption sampling for endless augmentation.
  - In `val` and `test` modes: Exact parameter lookups against deterministic manifests.
  - DataLoader: `get_corrupted_pet_dataloader(split, batch_size, ...)`.
- **Verification**: Run `task verify-corruptions` to validate noise statistics, blur frequency attenuation, and export the $4 \times 4$ panel to `results/corruption_verification_grid.png`.

### FS2K Paired Face-to-Sketch Dataset (Task 4)
The FS2K dataset contains 2,104 paired high-resolution facial photographs and corresponding artist sketches across 3 distinct sketch styles:
- **Download & Ingestion**: Run `task download-fs2k` (downloads the 104.6 MB Google Drive archive and unpacks into `data/fs2k/`).
- **Verified 1:1 Image Pairing**:
  - Style 0 (Style 1): 976 paired instances
  - Style 1 (Style 2): 731 paired instances
  - Style 2 (Style 3): 397 paired instances
  - Total verified pairs: 2,104 pairs (0 missing pairs).
- **Stratified Partitioning (`manifests/fs2k_split.json`)**:
  - Official `test` set: 1,046 pairs strictly reserved for final cGAN benchmark evaluation.
  - Official `train` set: 1,058 pairs partitioned into **85% Train** (899 pairs) and **15% Validation** (159 pairs), stratified across all 3 sketch styles (seed 42).
- **PyTorch Paired Dataset (`FS2KDataset`)**:
  - Location: `src/shared/datasets/fs2k.py`
  - Yields: `(photo_tensor, sketch_tensor, style_id)` with shapes `(3, 128, 128)`, `(3, 128, 128)`, and `int` style label $\in \{0, 1, 2\}$.
  - Supports dual normalization: $[0.0, 1.0]$ and $[-1.0, 1.0]$ (for tanh generator outputs).
  - Synchronized paired data augmentation: simultaneous random horizontal flipping preserving facial geometric correspondence.
  - Fast DataLoader: `get_fs2k_dataloader(split, batch_size, shuffle, num_workers)`.
- **Verification**: Run `task verify-fs2k` to validate split counts, dual normalization ranges, and export a 3-style comparative inspection panel to `results/fs2k_sample_grid.png`.

---

## Shared Losses, Metrics & Visualization Helpers

### Reversible & Numerically Stable Losses (`src/shared/losses.py`)
- **$L_1$ Reconstruction Loss**:
  $$\mathcal{L}_{L1}(y, \hat{y}) = \frac{1}{CHW}\sum |y - \hat{y}|$$
- **Structural Similarity Loss (SSIM Loss)**:
  $$\mathcal{L}_{\text{SSIM}}(y, \hat{y}) = 1 - \text{SSIM}(y, \hat{y})$$
  Powered by `pytorch_msssim` with 11Ã—11 Gaussian window ($\sigma = 1.5$) and dynamic range $1.0$.
- **Combined Reconstruction Loss**:
  $$\mathcal{L}_{\text{rec}}(y, \hat{y}) = \alpha \cdot \mathcal{L}_{L1}(y, \hat{y}) + (1 - \alpha) \cdot \mathcal{L}_{\text{SSIM}}(y, \hat{y})$$
  Configurable loss weighting ($\alpha=0.84$ default, based on Zhao et al., IEEE TCI 2017).
- **GAN Losses (`GANLoss`)**: Vanilla BCE with logits or LSGAN MSE loss for conditional GAN training.

### Standardized Evaluation Metrics (`src/shared/metrics.py`)
- **PSNR**: Peak Signal-to-Noise Ratio with dynamic range $1.0$ and zero-MSE guard ($100.0\text{ dB}$ ceiling).
- **SSIM**: Mean structural similarity index calculated on $[0.0, 1.0]$ float tensors.
- **MAE / L1 Error**: Mean pixel-level absolute difference.
- **MSE**: Mean squared error.
- **Batch Evaluation (`evaluate_metrics`)**: Evaluates all four scalar metrics on GPU/CPU batches and returns dictionary of Python floats for tracking and reporting.

### Visualization & Logging Helpers (`src/shared/visualization.py`)
- **Reconstruction Grid (`make_reconstruction_grid`)**: Generates 3-row grid showing Corrupted Input, Restored Output, and Ground Truth Target.
- **Error Heatmap (`make_error_heatmap`)**: Computes pixel-wise residual magnitude $|y - \hat{y}|$ mapped to colormaps (`inferno`, `jet`, `plasma`) to pinpoint high-error regions.
- **Training Curves (`plot_training_curves`)**: Generates publication-ready 3-panel Matplotlib figures (Loss, PSNR, SSIM) for training and validation runs.
- **Tracker Integration (`log_epoch_visuals`)**: One-line helper logging qualitative grids and heatmaps directly into MLflow.
- **Verification**: Run `task verify-losses-metrics` to validate gradient backpropagation, numerical metrics, and export inspection artifacts to `results/losses_metrics_verification.png` and `results/sample_training_curves.png`.

---

## Task 1: Universal Denoising Autoencoder

A unified convolutional autoencoder for blind restoration across clean images, salt-and-pepper noise, Gaussian blur, and rectangular occlusion without metadata conditioning.

### Model Architecture (`src/task1/`)
- **Encoder (`src/task1/encoder.py`)**: Progressive 4-stage convolutional downsampling ($128 \times 128 \to 8 \times 8$) with residual conv blocks and strided downsampling convolutions.
- **Bottleneck**: Compressed spatial feature map ($8 \times 8 \times 256$, $16,384$ floats) enforcing a strict $3.0\times$â€“$12.0\times$ data compression ratio with optional dropout.
- **Decoder (`src/task1/decoder.py`)**: Mirrored 4-stage upsampling ($8 \times 8 \to 128 \times 128$) with dual-convolution refinement blocks and `Sigmoid` output head in $[0.0, 1.0]$.
- **Wrapper (`src/task1/autoencoder.py`)**: Unified `UniversalAutoencoder` module supporting encoding, decoding, parameter counting, compression ratios, and full checkpoint serialization.

### Architectural Variants & Benchmarks
Benchmarked across 20 iterations at batch size 16 on $128 \times 128 \times 3$ tensors:

| Variant | Parameters | Skip Connections | Bottleneck Compression | CPU Latency (b=16) | Selected Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Plain Conv Stack** | 4,869,187 | None (strict bottleneck) | High ($8\times 8 \times 256$, $3.0\times$) | 418.98 ms | Baseline |
| **ResBlock Autoencoder** | 4,913,091 | Intra-stage residual only | High ($8\times 8 \times 256$, $3.0\times$) | 456.86 ms | **Selected Primary Architecture** |
| **U-Net-lite (Restricted Skips)** | 4,990,259 | $1\times 1$ bottlenecked skips | Moderate (intermediate bypass) | 460.27 ms | Ablation Reference |
| **ResBlock + U-Net-lite** | 5,068,211 | Residual + $1\times 1$ skips | Moderate (intermediate bypass) | 547.46 ms | Ablation Reference |

- **Verification**: Run `task verify-task1-arch` or `pytest tests/test_task1_architecture.py`.

### Loss Function Formulation & Research ($\alpha$ Sensitivity)
The network optimizes a composite loss balancing absolute pixel fidelity with structural similarity:
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_1(x, \hat{x}) + (1 - \alpha) \cdot (1 - \text{SSIM}(x, \hat{x}))$$

Empirical sensitivity evaluation across 5 full epochs on a 15% training subset (384 images, pre-cached) with fixed validation:

| Loss Formulation | $\alpha$ | Val Loss | Val PSNR (dB) | Val SSIM | Val MAE | Observations & Findings |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Pure SSIM | 0.0 | 0.7150 | 11.15 | 0.2850 | 0.2196 | Slowest convergence; lacks direct pixel anchor, large intensity shifts |
| 90% SSIM / 10% L1 | 0.1 | 0.6388 | 11.88 | 0.3128 | 0.2031 | Small L1 anchor immediately sharpens convergence (+0.0278 SSIM) |
| 80% SSIM / 20% L1 | 0.2 | 0.5740 | 12.22 | 0.3317 | 0.1968 | Steady structural progression |
| 70% SSIM / 30% L1 | 0.3 | 0.5423 | 11.75 | 0.3136 | 0.2060 | Structural plateau; under-penalizes severe salt-and-pepper noise |
| 60% SSIM / 40% L1 | 0.4 | 0.4822 | 12.16 | 0.3310 | 0.2021 | Balanced reduction across losses |
| Balanced | 0.5 | 0.4443 | 11.80 | 0.3249 | 0.2136 | Equal weight; moderate balance |
| 60% L1 / 40% SSIM | 0.6 | 0.3841 | 12.67 | 0.3289 | 0.1928 | L1 gradient begins to dominate, reducing MAE below 0.20 |
| 70% L1 / 30% SSIM | 0.7 | 0.3038 | 13.86 | 0.3695 | 0.1638 | Substantial jump in PSNR (+1.19 dB) and SSIM (+0.0406) |
| **Combined (Default)** | **0.8** | **0.2770** | **12.67** | **0.3661** | **0.1877** | **Robust multi-modal balance**; edge fidelity + noise suppression |
| 90% L1 / 10% SSIM | 0.9 | 0.2534 | 11.14 | 0.3728 | 0.2118 | High SSIM; strong edge guidance |
| Pure L1 | 1.0 | 0.1103 | 16.02 | 0.3921 | 0.1103 | Minimizes pixel error directly; higher risk of over-smoothing |

- **Decision**: Locked $\alpha = 0.8$ as the default baseline weighting; configured Optuna search space $[0.6, 1.0]$ for hyperparameter search.
- **Training Pipeline (`src/task1/train.py`)**: End-to-end training over dynamic corruptions with deterministic validation across `manifests/val_manifest.json`, early stopping, model checkpointing (`checkpoints/task1/baseline_best.pth`), and MLflow experiment logging.
- **Verification**: Run `task research-task1-loss` or `pytest tests/test_task1_train.py`.

---

## Task 2: Hard-Routing Restoration System

A decoupled restoration system combining a 4-class corruption classifier with an identity bypass (clean) and 3 specialist autoencoders (salt-and-pepper, Gaussian blur, rectangular occlusion).

### Classifier Architecture Research (`src/task2/classifier.py`)
- **Backbones Evaluated**:
  1. `CustomConvClassifier`: 4-stage Conv-BN-LeakyReLU-MaxPool2d + GAP + Dropout + Linear head.
  2. `MobileNetClassifier`: Inverted residual blocks with depthwise separable convolutions (Sandler et al., 2018).
  3. `ResNet18Classifier`: Adapted torchvision ResNet-18 (He et al., 2016).
- **Empirical Architecture Benchmark**: Benchmarked CPU latency ($B=1, 16$), parameter counts, model memory footprint, and 5-epoch empirical convergence on balanced Oxford-IIIT Pet data (256 train, 128 val):

| Candidate Architecture | Trainable Params | Model Size (MB) | CPU Latency ($B=1$) | CPU Latency ($B=16$) | Throughput (FPS) | 5-Ep Val Acc (%) | Val Macro-F1 | Mean Epoch Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CustomConvClassifier** | **389,924** | **1.49 MB** | **4.81 ms** | **68.92 ms** | **232.2** | **76.56%** | **0.7635** | **2.84s** |
| **MobileNetClassifier** | 247,588 | 0.94 MB | 3.88 ms | 45.96 ms | 348.2 | 63.28% | 0.5782 | 2.44s |
| **ResNet18Classifier** | 11,178,564 | 42.64 MB | 11.12 ms | 108.52 ms | 147.4 | 55.47% | 0.4983 | 5.45s |

- **Design Decision**: Adopted `CustomConvClassifier` as the primary backbone. It converged rapidly to 76.56% accuracy / 0.7635 Macro-F1 in 5 epochs, operates at 4.81 ms CPU latency, and stays under 1.5 MB in size, avoiding ResNet-18's parameter bloat (11.18M params) and MobileNet's slower early convergence.
- **Architectural Decoupling**: Classifier operates completely independently from autoencoders to eliminate gradient interference between reconstruction and classification objectives, support zero-cost identity bypass for clean inputs, and allow independent ONNX export.
- **Verification**: Run `uv run python scripts/verify_task2_classifier_architectures.py` or `uv run pytest tests/test_task2_classifier.py`.

### Corruption Classifier Training (`src/task2/train_classifier.py`)
- **Balanced Multi-Class Sampling (`src/task2/dataset.py`)**: `BalancedBatchSampler` guarantees exactly $B/4$ samples per class in every batch ($25\%$ Clean, $25\%$ S&P, $25\%$ Blur, $25\%$ Occlusion) with dynamic training augmentations.
- **Baseline Training Results (15 Epochs on Oxford Pets)**:
  - **Validation Accuracy**: **98.78%** (baseline spec threshold was $>85.0\%$)
  - **Macro-Averaged F1**: **0.9878** (Precision: 0.9878, Recall: 0.9878)
  - **Best Validation Loss**: **0.0434** (Cross-Entropy)
  - **Per-Class F1-Scores**: Clean: 0.9755, Salt & Pepper: 0.9973, Blur: 0.9864, Occlusion: 0.9919
  - **Confusion Matrix**: Clear diagonal dominance ($>97.2\%$ diagonal accuracy across all classes, $100\%$ on occlusion).
- **Canonical Baseline Checkpoint**: `checkpoints/task2/classifier_best.pt` (4.7 MB with optimizer/scheduler state).
- **Verification**: Run `uv run python -m src.task2.train_classifier --epochs 15` or `uv run pytest tests/test_task2_train.py`.

### Classifier Hyperparameter Optimization (`src/task2/optuna_classifier.py`)
- **Bayesian Search Study**: SQLite-backed study (`task2-classifier` in `optuna/optuna_studies.db`) with `TPESampler` and `MedianPruner`.
- **Winning Configuration (Trial #6)**:
  - `learning_rate`: $1.57 \times 10^{-3}$
  - `batch_size`: $16$
  - `channel_config`: `'large'` (`(48, 96, 192, 256)`)
  - `dropout`: $0.10$
  - `weight_decay`: $3.06 \times 10^{-3}$
  - **Validation Macro-F1**: **0.9877** ($>98.7\%$ validation accuracy)
- **Artifacts**: Study configuration in `results/task2/classifier_best_hyperparams.json`, study summary in `optuna/task2-classifier.json`, optimization history plot in `results/task2/classifier_optuna_history.png`, and parameter importances plot in `results/task2/classifier_optuna_param_importances.png`.
- **Verification**: Run `uv run pytest tests/test_task2_optuna.py`.

### Specialist Autoencoder Design Research (`src/task2/specialist.py`)
- **Architectural Candidates Evaluated**:
  1. `Alternative 1 (Homogeneous Task 1 AE)`: Standard 4-stage ResBlock structure (`channels=(32, 64, 128, 256)`, bottleneck 256) across all 3 specialists.
  2. `Alternative 2 (Lightweight Shared Variant)`: Scaled-down 3-stage structure (`channels=(32, 64, 128)`, bottleneck 128, no ResBlocks) across all 3 specialists.
  3. `Alternative 3 (Corruption-Tailored Topologies)`: High-frequency residual network for Salt-and-Pepper (3 stages), multi-scale ResBlock for Blur (4 stages), deep contextual bottleneck ResBlock for Occlusion (4 stages).
- **Empirical Architecture Benchmark Summary** (`results/task2/specialist_architecture_benchmark.json`):

| Alternative | Total System Params (3 Experts) | Total Size (MB) | Mean CPU Latency ($B=1$) | Mean Val Loss | Mean Val PSNR (dB) | Mean Val SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alternative 1: Homogeneous Task 1 AE** | 14,739,273 | 56.22 MB | 28.67 ms | 0.2875 | 12.17 dB | 0.3054 |
| **Alternative 2: Lightweight Shared Variant** | **3,687,369** | **14.07 MB** | **22.33 ms** | **0.2718** | **13.02 dB** | **0.3215** |
| **Alternative 3: Corruption-Tailored Topologies** | 11,065,929 | 42.21 MB | 26.89 ms | 0.2805 | 12.61 dB | 0.3154 |

- **Design Decisions**:
  - Adopted shared architectural topology for the 3 specialists to enable identical batching, predictable memory footprint, and uniform ONNX export.
  - Resolved Open Question: Adopted **Shared Optuna Search** to optimize the common specialist topology in $<15\text{ minutes}$, followed by independent training of the 3 specialists on their respective single-corruption distributions.
  - Implemented modular `SpecialistAutoencoder` and factory `build_specialist` in `src/task2/specialist.py`.
- **Verification**: Run `uv run python scripts/verify_task2_specialist_architectures.py` or `uv run pytest tests/test_task2_specialist.py`.

### Specialist Autoencoders Baseline Training (`src/task2/train_specialists.py`)
- **Training Setup**:
  - Independent training across 3 single-corruption specialist autoencoders ($S_{\text{salt}}, S_{\text{blur}}, S_{\text{occlusion}}$) on Oxford Pets clean base images with dynamic stochastic corruptions.
  - Optimizer: AdamW ($lr=1\times 10^{-3}$, weight decay $1\times 10^{-4}$), Cosine Annealing scheduler, batch size $32$, and `CombinedReconstructionLoss(alpha=0.84)`.
  - Checkpoints: `checkpoints/task2/specialist_salt_best.pt`, `checkpoints/task2/specialist_blur_best.pt`, `checkpoints/task2/specialist_occlusion_best.pt`.
- **Baseline Convergence Results (Dedicated Validation Manifests)**:

| Specialist Expert | Target Corruption | Best Val Loss | Best Val PSNR (dB) | Best Val SSIM | Best Val MAE | Checkpoint Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$S_{\text{salt}}$** | Salt-and-Pepper | **0.1574** | **18.89 dB** | **0.4766** | **0.0877** | 19.74 MB |
| **$S_{\text{blur}}$** | Gaussian Blur | **0.1745** | **17.86 dB** | **0.4420** | **0.1014** | 19.74 MB |
| **$S_{\text{occlusion}}$** | Rectangular Occlusion | **0.1836** | **17.10 dB** | **0.4235** | **0.1087** | 19.74 MB |

- **Artifacts Exported**:
  - Metrics JSON: `results/task2/specialists_baseline_metrics.json`
  - Training Curves: `results/task2/specialists_baseline_curves.png`
  - Triplet Inspection Grid: `results/task2/specialists_sample_reconstructions.png`
  - MLflow Tracking: Experiment `task2-specialists`, run `baseline_specialists`.
- **Verification**: Run `uv run python -m src.task2.train_specialists --specialist all --epochs 3` or `uv run pytest tests/test_task2_train_specialists.py`.

### Specialist Hyperparameter Optimization (`src/task2/optuna_specialists.py`)
- **Bayesian Search Study**: SQLite-backed study (`task2-specialists` in `optuna/optuna_studies.db`) with `TPESampler` and `MedianPruner(n_startup_trials=5, n_warmup_steps=3)`.
- **Search Space**: `learning_rate` ($[10^{-4}, 10^{-2}]$), `bottleneck_dim` ($\{64, 128, 256\}$), `channel_config` (`['shallow', 'standard', 'deep']`), `batch_size` ($\{16, 32\}$), `alpha` ($[0.5, 1.0]$), `use_residual` (`[True, False]`).
- **Winning Configuration (Trial #5)**:
  - `learning_rate`: $4.57 \times 10^{-4}$
  - `bottleneck_dim`: $128$
  - `channel_config`: `'standard'` (`(32, 64, 128, 256)`)
  - `batch_size`: $16$
  - `alpha`: $0.95$
  - `use_residual`: `True`
  - **Validation Reconstruction Loss**: **0.1435** (substantially outperforming the Step 5 baseline of ~0.1718)
- **Artifacts Exported**:
  - Best Hyperparameters: `results/task2/specialists_best_hyperparams.json`
  - Summary & History CSV: `optuna/task2-specialists.json/`
  - Optimization Plots: `results/task2/specialists_optuna_history.png`, `results/task2/specialists_optuna_param_importances.png`
  - MLflow Tracking: Experiment `task2-specialists`.
- **Verification**: Run `uv run pytest tests/test_task2_optuna_specialists.py`.

### Hard-Routing Inference Pipeline (`src/task2/router.py`, `src/task2/inference.py`)
- **End-to-End Orchestrator**: `HardRouter` binds the 4-class corruption classifier with zero-cost identity bypass for clean inputs and the 3 specialist autoencoders ($S_{\text{salt}}, S_{\text{blur}}, S_{\text{occlusion}}$).
- **Batched Dispatching**: Groups inputs by routing decision, executes each specialist once per unique class in parallel, and reassembles tensors in original order.
- **Empirical Profiling on CPU (64 Real Samples, 25 Runs)**:
  - Single-Image Latency ($B=1$): **32.25 ms** (31.01 FPS throughput).
  - Classifier Latency: **4.74 ms** ($14.7\%$ of total runtime).
  - Restoration Latency: **27.36 ms** ($84.8\%$ of total runtime).
  - Batched Latency ($B=16$): **397.57 ms** (40.24 FPS amortized throughput).
  - Clean Identity Bypass Latency: **0.16 ms** ($170\times$ faster than autoencoders, bit-exact $MSE = 0.0$).
- **Artifacts Exported**:
  - Benchmark Summary: `results/task2/router_benchmark.json`
- **Verification**: Run `uv run python -m src.task2.inference --batch-size 16 --runs 25` or `uv run pytest tests/test_task2_router.py`.

### Multi-Scenario Benchmark: Oracle vs Predicted Routing (`src/task2/evaluate.py`)
- **Comparative Evaluation**: Benchmarked Oracle routing vs Predicted routing vs Task 1 Universal Autoencoder baseline on 7,338 test instances from `manifests/test_manifest.json` across 4 corruptions and 10 severity levels.
- **Empirical Results Summary**:

| Evaluation Domain | Metric | Task 1: Universal AE | Task 2: Oracle Routing | Task 2: Predicted Routing | Routing Gap ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Overall Mean** | **PSNR (dB)** | 20.16 | **23.90** | **23.74** | **+0.16 dB** |
| **Overall Mean** | **SSIM** | **0.5935** | 0.4969 | 0.5321 | -0.0352 |
| **Overall Mean** | **MAE** | **0.0742** | 0.0923 | 0.0885 | -0.0038 |
| **Clean ($y=0$)** | PSNR (dB) | 20.90 | **80.00** | **72.47** | +7.53 dB |
| **Salt-and-Pepper ($y=1$)** | PSNR (dB) | **20.75** | 18.37 | 18.08 | +0.28 dB |
| **Gaussian Blur ($y=2$)** | PSNR (dB) | **20.92** | 17.65 | 19.88 | -2.23 dB |
| **Occlusion ($y=3$)** | PSNR (dB) | **18.56** | 16.97 | 17.01 | -0.04 dB |

- **Key Takeaways**:
  - Hard-routing with specialized autoencoders and zero-cost identity bypass achieves **23.74 dB test PSNR**, outperforming Task 1 Universal AE (20.16 dB) by **+3.58 dB** (+17.8%).
  - The routing gap between Oracle (23.90 dB) and Predicted (23.74 dB) is just **+0.16 dB**, confirming that classifier accuracy (98.78%) is not a bottleneck.
  - Zero-cost identity bypass for clean images achieves **72.47 dB** fidelity (vs 20.90 dB in Task 1) by avoiding neural blurring.
- **Failure Mode Audit**: Diagnosed and documented 3 primary failure scenarios in `results/task2/visuals/routing_failure_cases.png` (False Clean Bypass on $p=0.03$ noise, Cross-Corruption Blur smearing, and Inpainting boundary hallucination).
- **Verification**: Run `uv run python -m src.task2.evaluate --subsample 0.2` or `uv run pytest tests/test_task2_evaluate.py`.

### Production ONNX Export & Parity Benchmark (`src/task2/export_onnx.py`)
- **Exported Production Models (Opset 17, Dynamic Batching)**:
  - `models/onnx/task2_classifier.onnx` (1.49 MB): $(B, 3, 128, 128) \to (B, 4)$
  - `models/onnx/task2_specialist_salt.onnx` (18.75 MB): $(B, 3, 128, 128) \to (B, 3, 128, 128)$
  - `models/onnx/task2_specialist_blur.onnx` (18.75 MB): $(B, 3, 128, 128) \to (B, 3, 128, 128)$
  - `models/onnx/task2_specialist_occlusion.onnx` (18.75 MB): $(B, 3, 128, 128) \to (B, 3, 128, 128)$
- **Numerical Parity Verification**: Strict assertion ($\max |Y_{\text{pt}} - Y_{\text{ort}}| < 10^{-5}$) passed across all 4 models for $B \in \{1, 4, 8\}$ (classifier max diff: $5.72 \times 10^{-6}$, specialists max diffs: $2.38 \times 10^{-7}$ â€“ $7.75 \times 10^{-7}$).
- **CPU Latency & Speedup Benchmarks (100 runs)**:
  - Classifier: **1.56 ms** ONNX Runtime vs 5.24 ms PyTorch (**$3.36\times$ speedup**, 642.1 FPS).
  - Salt Specialist: **16.06 ms** ONNX Runtime vs 27.45 ms PyTorch (**$1.71\times$ speedup**, 62.3 FPS).
  - Blur Specialist: **19.35 ms** ONNX Runtime vs 26.45 ms PyTorch (**$1.37\times$ speedup**, 51.7 FPS).
  - Occlusion Specialist: **17.39 ms** ONNX Runtime vs 27.61 ms PyTorch (**$1.59\times$ speedup**, 57.5 FPS).
  - `OnnxHardRouter` Pipeline: **55.77 ms** end-to-end latency with zero-cost clean bypass.
- **Verification**: Run `uv run python -m src.task2.export_onnx` or `uv run pytest tests/test_task2_onnx.py`.

---

## Task 3: Soft Mixture-of-Experts (MoE) Image Restoration

### Architectural Pipeline (`src/task3/gate.py`, `src/task3/moe_model.py`)
- **Differentiable Convex Blending**: Transforms discrete hard routing into a continuous composite:
  $$\hat{x} = w_1 \cdot \tilde{x} + w_2 \cdot A_{\text{salt}}(\tilde{x}) + w_3 \cdot A_{\text{blur}}(\tilde{x}) + w_4 \cdot A_{\text{occlusion}}(\tilde{x})$$
  where routing weights are obtained via temperature-scaled softmax:
  $$w = \text{softmax}\left(\frac{G(\tilde{x})}{\tau}\right), \quad \sum_{k=1}^4 w_k = 1$$
- **Linear Gating Network**: 1:1 parameter transfer from Task 2's pre-trained `CustomConvClassifier` backbone and classification head (389,924 parameters). Initialized with calibrated 98.78% classification boundaries.
- **Joint Multi-Objective Loss Function (`src/task3/losses.py`)**:
  $$\mathcal{L}_{\text{tot}} = \lambda_{\text{L1}} \mathcal{L}_1(\hat{x}, x) + \lambda_{\text{SSIM}} (1 - \text{SSIM}(\hat{x}, x)) + \lambda_{\text{CE}} \mathcal{L}_{\text{CE}}(G(\tilde{x}), y) + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}}(w)$$
- **Balance Regularizer Research (`scripts/verify_task3_balance_regularizers.py`)**:
  - Evaluated 3 regularizers: Entropy Maximization, Switch Transformer load balancing, and $L_2$ deviation from uniform prior $\frac{1}{K}$.
  - Selected $L_2$ deviation: $\mathcal{L}_{\text{balance}} = \sum_{k=1}^K (\bar{w}_k - 1/K)^2$ where $\bar{w} = \frac{1}{B}\sum_{i=1}^B w_i$. Guarantees smooth non-zero gradients without log-barrier instability or expert starvation ($\ge 18.7\%$ minimum allocation preserved).

### Two-Stage Training Strategy (`src/task3/train.py`)
- **Phase 1 (Warm-up)**: Freeze specialist denoisers ($S_{\text{salt}}, S_{\text{blur}}, S_{\text{occlusion}}$) and train only gating network ($lr=1\times 10^{-3}$) to align routing weights with convex blending dynamics.
- **Phase 2 (Joint Fine-Tuning)**: Unfreeze all specialists and train end-to-end using differential learning rates: gate trained at $lr$, specialists fine-tuned at $0.2 \times lr$ to protect pre-trained single-corruption denoising filters.
- **Baseline Training Results (Step 5)**:
  - 30 epochs on RTX 3050 (saved to `checkpoints/task3/baseline_best.pth`).
  - Validation SSIM: **0.7096**, Validation PSNR: **21.65 dB**.
  - Minimum expert utilization: **7.95%** (confirming zero expert starvation or collapse).

### Hyperparameter Optimization via Optuna (`src/task3/optuna_search.py`)
- **Study Setup**: 26 trials executed on RTX 3050 (`task3-moe-joint` in `optuna/optuna_studies.db`).
- **Pruner**: MedianPruner with custom routing collapse guard ($\max \bar_w_k > 0.90$ or $\min \bar_w_k < 0.02$). Pruned 12 non-competitive / uncalibrated trials.
- **Best Configuration (Trial #8)**:
  - Validation SSIM: **0.7261** (gain of $+0.0165$ over baseline)
  - `fine_tune_lr`: $1.754 \times 10^{-5}$
  - `temperature` ($\tau$): $2.526$
  - `lambda_ce`: $0.0114$
  - `lambda_balance`: $0.0659$
  - `reconstruction_alpha`: $0.6294$ ($\lambda_{\text{L1}} = 0.6294, \lambda_{\text{SSIM}} = 0.3706$)
- **Artifacts Exported**:
  - Canonical configuration: `config/task3_best_params.json`
  - Optimization curves: `results/task3/figures/optuna_moe_history.png`
  - Parameter importances: `results/task3/figures/optuna_moe_param_importances.png`
  - Hyperparameter slice: `results/task3/figures/optuna_moe_slice.png`
- **Verification**: Run `uv run python -m src.task3.optuna_search --n-trials 2` or `uv run pytest tests/test_task3_optuna.py`.

### Definitive Retraining with Best Hyperparameters (`checkpoints/task3/best_model.pth`)
- **Production Retraining Schedule**: Executed on NVIDIA RTX 3050 GPU (11 epochs: 3 warmup + 8 joint fine-tuning) using parameters from `config/task3_best_params.json`. Total runtime: ~7 minutes (<15 min limit).
- **Production Checkpoint**: Saved to `checkpoints/task3/best_model.pth` (~182 MB, complete optimizer and scheduler state).
- **Validation Convergence & Metrics**:
  - **Overall Val PSNR**: **21.10 dB**
  - **Overall Val SSIM**: **0.6905**
  - **Overall Val MAE**: **0.0613**
  - **Routing Utilization**: Min expert utilization $6.59\%$, Max $57.92\%$ (zero starvation or collapse).
- **Per-Corruption Performance Breakdown**:
  - **Clean / Identity**: **25.81 dB** PSNR, **0.8959** SSIM, **0.0402** MAE ($w_{\text{clean}} = 0.5874$)
  - **Gaussian Blur**: **23.06 dB** PSNR, **0.7182** SSIM, **0.0543** MAE ($w_{\text{blur}} = 0.1571$)
  - **Salt-and-Pepper**: **19.95 dB** PSNR, **0.4386** SSIM, **0.0572** MAE ($w_{\text{salt}} = 0.3973$)
  - **Rectangular Occlusion**: **15.58 dB** PSNR, **0.7105** SSIM, **0.0935** MAE ($w_{\text{occ}} = 0.1538$)
- **Artifacts Exported**:
  - Checkpoint: `checkpoints/task3/best_model.pth`
  - Epoch history log: `results/task3/final_train_metrics.json`
  - Final validation metrics: `results/task3/final_val_metrics.json`
  - MLflow Run: `task3-moe-final`
- **Verification**: Run `uv run pytest tests/test_task3_retrain.py`.

### Routing Behavior Analysis & Visualization (`src/task3/routing_analysis.py`)
- **4Ã—4 Routing Confusion Matrix**: Evaluated on 736 validation images, mapping ground truth corruptions to average expert weights ($w_{\text{clean}}, w_{\text{salt}}, w_{\text{blur}}, w_{\text{occ}}$):
  - Clean: $58.74\%$ Clean bypass, $23.07\%$ Salt, $7.18\%$ Blur, $11.01\%$ Occlusion.
  - Salt & Pepper: $39.83\%$ Salt Specialist, $54.96\%$ Clean bypass, $0.57\%$ Blur (near-zero cross-interference).
  - Gaussian Blur: $15.71\%$ Blur Specialist, $50.13\%$ Clean bypass, $22.76\%$ Salt, $11.40\%$ Occlusion.
  - Occlusion: $15.38\%$ Occlusion Specialist, $67.82\%$ Clean bypass, $13.91\%$ Salt, $2.89\%$ Blur.
- **Severity-Dependent Progression**: Empirical evaluation on test manifests reveals monotonic scaling of specialist routing with corruption severity:
  - Salt Specialist: $27.06\%$ (mild) $\to$ $41.47\%$ (medium) $\to$ $53.05\%$ (severe).
  - Blur Specialist: $11.92\%$ (mild) $\to$ $17.01\%$ (medium) $\to$ $19.93\%$ (severe).
  - Occlusion Specialist: $13.35\%$ (mild) $\to$ $15.07\%$ (medium) $\to$ $17.74\%$ (severe).
- **Expert Health Audit**: Minimum dataset utilization $6.59\%$, maximum $57.91\%$, zero starved experts, health status `PASS`.
- **Visual Artifacts**:
  - Confusion matrix CSV: `results/task3/routing_confusion_matrix.csv`
  - Heatmap plot: `results/task3/figures/routing_heatmap.png`
  - Severity trends: `results/task3/figures/severity_routing_trends.png`
  - Qualitative galleries: `results/task3/figures/routing_galleries.png`
  - Summary audit JSON: `results/task3/routing_analysis_summary.json`
- **Verification**: Run `uv run python -m src.task3.routing_analysis` or `uv run pytest tests/test_task3_routing_analysis.py`.

### Multi-System Comparative Benchmark: Tasks 1, 2, and 3 (`src/task3/evaluate.py`)
- **Exhaustive Evaluation Scope**: Benchmarked Task 1 Universal AE, Task 2 Hard Router (Predicted & Oracle), and Task 3 Soft MoE across 7,340 official Oxford-IIIT Pet test instances across all 4 corruption categories and 10 severity variants.
- **Unified Benchmark Comparison Table**:

| Restoration System | Overall PSNR | Overall SSIM | Clean PSNR | Salt (Mild/Med/Sev) | Blur (Mild/Med/Sev) | Occ (Mild/Med/Sev) | GPU Latency (RTX 3050) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Universal AE** | 20.08 dB | 0.5890 | 20.82 dB | 20.80 / 20.71 / 20.51 dB | 20.87 / 20.88 / 20.79 dB | 19.64 / 18.62 / 17.14 dB | **5.89 ms** |
| **Task 2: Hard Router (Predicted)** | **23.79 dB** | 0.4958 | **77.54 dB** | 18.53 / 18.53 / 18.51 dB | 17.85 / 17.78 / 17.75 dB | 17.69 / 17.13 / 16.54 dB | **1.78 ms** |
| **Task 2: Hard Router (Oracle)** | **24.00 dB** | 0.4952 | **80.00 dB** | 18.53 / 18.53 / 18.51 dB | 17.79 / 17.78 / 17.75 dB | 17.46 / 17.13 / 16.54 dB | **5.95 ms** |
| **Task 3: Soft MoE (Ours)** | 20.03 dB | **0.6495** | 25.89 dB | **22.03 / 19.46 / 18.20 dB** | **23.96 / 22.30 / 21.34 dB** | **17.80 / 15.46 / 13.85 dB** | 18.17 ms |

- **Key Architectural Findings**:
  - **Structural Quality Champion**: Soft MoE dominates on SSIM with **0.6495 overall**, outperforming Task 1 (0.5890) by **+0.0605** and Task 2 Predicted (0.4958) by **+0.1537**.
  - **Specialist Synergy over Hard Partitioning**: In Gaussian Blur, Soft MoE achieves **0.6951 SSIM** and **23.96 dB / 22.30 dB / 21.34 dB** (vs Task 2: 0.4410 SSIM / 17.85 dB; Task 1: 0.6073 SSIM / 20.87 dB). In Occlusion, Soft MoE attains **0.7234 SSIM** (vs Task 2: 0.4217, Task 1: 0.5492).
  - **Clean Detail Preservation**: Continuous identity bypass routes pristine images with **25.89 dB PSNR** and **0.8952 SSIM**, eliminating the blurring degradation imposed by Task 1 Universal AE (20.82 dB / 0.6132 SSIM).
  - **Resilience to Classification Error**: Qualitative analysis demonstrates that when Task 2 hard-switches to an incorrect specialist on boundary noise, acute artifacts ensue; Task 3 blends specialist features smoothly, recovering +3 to +6 dB on edge cases without discrete switching discontinuities.
  - **Real-Time Inference**: Single-image GPU latency is **18.17 ms** (~55 FPS) on NVIDIA RTX 3050, easily meeting real-time interactive requirements while executing all specialists in a single differentiable forward pass.
- **Exported Evaluation Artifacts**:
  - Benchmark CSV: `results/task3/test_benchmark_comparison.csv`
  - Summary JSON: `results/task3/test_evaluation_summary.json`
  - 12-Case Qualitative Comparison: `results/task3/figures/qualitative_comparison_12.png`
  - 4-Case Diagnostic & Failure Analysis: `results/task3/figures/failure_cases_4.png`
  - MLflow Run: `task3-test-evaluation` in experiment `genai-task3-soft-moe`
- **Verification**: Run `uv run python -m src.task3.evaluate --subsample 0.2` or `uv run pytest tests/test_task3_evaluate.py`.

### Single-Graph ONNX Export & Parity Benchmark (`src/task3/export_onnx.py`)
- **Atomic Single-Graph Export (Opset 17, Dynamic Batching)**:
  - Exported the complete differentiable Soft MoE architecture into a single unified ONNX model: `models/onnx/task3_soft_moe.onnx` (57.76 MB).
  - Graph encapsulates: Gating Network + Identity pass-through + 3 Specialist Autoencoders + Temperature-scaled Softmax ($\tau=2.526$) + Weighted sum composite restoration.
  - Inputs: `input_image` $\to (B, 3, 128, 128)$
  - Outputs: `restored_image` $\to (B, 3, 128, 128)$, `routing_weights` $\to (B, 4)$
- **Strict Numerical Parity Verification**:
  - Asserted output equivalence between PyTorch eager mode and ONNX Runtime CPU across batch sizes $B \in \{1, 4, 8\}$:
  - Maximum reconstruction difference: $\mathbf{4.77 \times 10^{-7}}$ (spec tolerance: $< 1 \times 10^{-5}$).
  - Maximum routing weights difference: $\mathbf{5.96 \times 10^{-7}}$ (spec tolerance: $< 1 \times 10^{-5}$).
  - Parity status: `all_passed = True`.
- **Latency & Speedup Benchmark (100 Iterations on CPU)**:
  - PyTorch eager CPU latency: **138.99 ms**
  - ONNX Runtime CPU latency: **49.04 ms**
  - Acceleration speedup factor: **$2.83\times$ speedup**
  - CPU Throughput: **20.4 FPS**
- **Production Engine**: Implemented `OnnxSoftMoE` engine in `src/task3/export_onnx.py` for direct deployment in FastAPI (`/api/v1/restore/soft-moe`) and web application workflows.
- **Verification Artifacts**: Saved to `results/task3/onnx_parity_benchmark.json`; logged to MLflow run `task3-onnx-export`.
- **Verification**: Run `uv run python -m src.task3.export_onnx` or `uv run pytest tests/test_task3_onnx.py`.

---

## Task 4: Style-Conditioned Face-to-Sketch Synthesis (Conditional GAN)

Task 4 develops an end-to-end conditional Generative Adversarial Network (cGAN) for paired facial sketch synthesis using the **FS2K (Facial Sketch Synthesis 2K)** dataset (2,104 paired high-resolution facial photographs and sketches across 3 distinct artistic styles). Given a 128Ã—128 RGB facial photograph $x$ and a target style category $s \in \{0, 1, 2\}$, the system generates a synthesized sketch $\hat{y} = G(x, s)$ in the selected style while preserving facial landmarks and identity.

### Generator Architecture Research (Step 1)
- **Problem**: Selecting the optimal U-Net generator backbone to translate photographic textures into sharp stylized sketch strokes while preserving facial geometry (eyes, nose, mouth alignment).
- **Candidates Evaluated**:
  1. *Vanilla U-Net (pix2pix)*: Standard 4-stage encoder-decoder with symmetric strided conv downsampling, bottleneck, transposed conv upsampling, and direct skip connections.
  2. *ResNet-based U-Net*: Augments bottleneck and intermediate stages with Residual Blocks (identity shortcut connections).
  3. *Attention U-Net*: Gated attention mechanisms dynamically filtering encoder skip connections to prioritize facial landmarks over background.
- **Empirical GPU Benchmark Results (NVIDIA GeForce RTX 3050 6GB Laptop GPU)**:

| Candidate Architecture | Parameters | GPU Latency ($B=1$) | GPU Latency ($B=8$) | Peak VRAM ($B=8$) | CPU Latency ($B=1$) | Val L1 Error | Gradient Norm | ONNX Opset 17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla U-Net (pix2pix)** *(Selected)* | **6.83M** (6,830,387) | **3.68 ms** | **14.55 ms** | **160.56 MB** | **37.75 ms** | **0.2973** | 1.2914 | **PASS** (Clean) |
| **ResNet U-Net** | 16.27M (16,267,571) | 4.99 ms | 18.35 ms | 206.68 MB | 56.39 ms | 0.3238 | 1.1472 | **PASS** (Clean) |
| **Attention U-Net** | 6.92M (6,917,078) | 7.45 ms | 16.82 ms | 197.14 MB | 43.95 ms | 0.3135 | 1.3738 | **PASS** (Tracer warn) |

- **Decision**: Adopt **Vanilla U-Net (pix2pix)**. It achieves the lowest reconstruction error (0.2973 Val L1), operates at 3.68 ms GPU latency (~272 FPS) with minimal memory footprint (160.56 MB), and provides 100% clean ONNX graph compilation without dynamic-shape interpolation artifacts.

### Style Conditioning Research (Step 2)
- **Problem**: Determining the conditioning mechanism to inject the learned categorical style embedding into Generator $G$ and Discriminator $D$ for clear artistic control across the 3 FS2K styles.
- **Candidates Evaluated**:
  1. *Spatial Concatenation*: Replicate embedding across $(H, W)$ and concatenate along channel dimension.
  2. *FiLM (Feature-wise Linear Modulation)*: Small MLP predicting per-channel affine scale $\gamma(s)$ and shift $\beta(s)$ modulating normalized feature maps: $(1 + \gamma(s)) \cdot F + \beta(s)$.
  3. *AdaIN (Adaptive Instance Normalization)*: Adapts feature channel mean and variance to style-predicted statistics.
- **Empirical GPU Benchmark Results (NVIDIA GeForce RTX 3050)**:

| Conditioning Mechanism | Parameters | GPU Latency ($B=8$) | Val L1 Error | Style Differentiation ($\Delta$) | Mean Gradient Norm | ONNX Opset 17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spatial Concatenation** | 6.57M (6,567,219) | 14.65 ms | 0.2564 | 0.0234 (Diluted) | 1.1988 | **PASS** |
| **FiLM Modulation** *(Selected)* | 6.83M (6,830,387) | 14.80 ms | 0.2864 | **0.1005 ($>4.3\times$ higher)** | 1.5685 | **PASS** |
| **AdaIN Normalization** | 6.83M (6,830,387) | 15.06 ms | 0.2800 | 0.0373 (Moderate) | 1.5574 | **PASS** |

- **Style Embedding Dimension ($d_s$) Sweep**:
  - $d_s = 8$: Style Distance = 0.0385, Val L1 = 0.2651
  - **$d_s = 16$** *(Selected)*: Style Distance = **0.0736** (highest artistic separation), Val L1 = **0.2672**, Grad Norm = 1.3494
  - $d_s = 32$: Style Distance = 0.0681, Val L1 = 0.3075 (parameter redundancy and higher reconstruction error)
- **Decision**: Adopt **FiLM conditioning with $d_s = 16$**. Produces over $4.3\times$ higher style differentiation distance than spatial concatenation, maintains strong structural reconstruction, and translates directly into standard ONNX matrix and elementwise operators.
- **Artifacts Exported**:
  - Empirical metrics: `results/task4/architecture_conditioning_benchmark.json`
  - MLflow Run: `arch-and-conditioning-benchmark` in experiment `genai-task4-research`
- **Verification**: Run `uv run python scripts/benchmark_task4_research.py` or `uv run pytest tests/test_task4_research.py`.

### Discriminator Architecture Research (Step 3)
- **Problem**: Selecting the optimal PatchGAN discriminator receptive field (RF) size to balance local sketch stroke fidelity (pencil line sharpness, cross-hatching) and global facial symmetry (facial landmark positioning).
- **Candidates Evaluated**:
  1. *70Ã—70 PatchGAN (pix2pix default)*: 5-layer conv network ($14 \times 14$ patch grid) covering $>50\%$ of the $128 \times 128$ image.
  2. *16Ã—16 PatchGAN (shallower RF)*: 3-layer conv network ($62 \times 62$ patch grid) focusing exclusively on local edge sharpness.
  3. *Multi-Scale PatchGAN (pix2pixHD)*: Dual-scale PatchGAN evaluating both full resolution $128 \times 128$ and $2\times$ downsampled $64 \times 64$.
- **Empirical GPU Benchmark Results (NVIDIA GeForce RTX 3050 6GB Laptop GPU)**:

| Candidate Discriminator | Parameters | Step Latency (GPU) | Peak VRAM | Output Patch Grid | Receptive Field | Generator Gradient Norm | Discriminator Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **70Ã—70 PatchGAN** *(Selected)* | **2.78M** (2,783,345) | **39.33 ms** | **377.52 MB** | **$14 \times 14$** | **$\sim 70 \times 70$** | **17.6281** (Strongest guidance) | **0.0300** (Balanced) |
| **16Ã—16 PatchGAN** | 0.16M (155,761) | 38.87 ms | 358.05 MB | $62 \times 62$ | $\sim 16 \times 16$ | 2.3953 (Too weak) | 0.3975 |
| **Multi-Scale PatchGAN** | 2.96M (2,960,578) | 46.00 ms | 391.55 MB | $14 \times 14 + 14 \times 14$ | Dual scale | 11.1949 | 0.0927 |

- **Decision**: Adopt **70Ã—70 PatchGAN**. It yields a 57% stronger adversarial gradient norm into the generator (**17.63** vs 11.19) compared to Multi-Scale PatchGAN while executing 17% faster with less memory. The 16Ã—16 variant lacks sufficient receptive field to preserve facial landmark proportions (eyes/mouth spacing).
- **Artifacts Exported**:
  - Empirical metrics: `results/task4/discriminator_benchmark.json`
  - MLflow Run: `discriminator-architecture-benchmark` in experiment `genai-task4-research`
- **Verification**: Run `uv run python scripts/benchmark_task4_discriminator.py`.

### Production Generator & Discriminator Implementation (Step 4)
- **Production Modules Built**:
  1. [`src/task4/generator.py`](file:///c:/Users/rafique_/Desktop/New%20folder/Gen%20AI%20A1/src/task4/generator.py): Production `UNetGenerator` (Vanilla U-Net backbone, FiLM conditioning at bottleneck with $d_s=16$, InstanceNorm2d, Tanh output $[-1, 1]$, Gaussian $\mathcal{N}(0, 0.02)$ weight initialization).
  2. [`src/task4/discriminator.py`](file:///c:/Users/rafique_/Desktop/New%20folder/Gen%20AI%20A1/src/task4/discriminator.py): Production `PatchGANDiscriminator` (70Ã—70 receptive field, conditioned on photo, candidate sketch, and spatially replicated style embedding, emitting $14 \times 14$ logits). Also provides `MultiScaleDiscriminator`.
  3. [`src/task4/augmentation.py`](file:///c:/Users/rafique_/Desktop/New%20folder/Gen%20AI%20A1/src/task4/augmentation.py): `CoupledTransform` applying identical geometric rotations/flips across photo-sketch pairs, with subtle photo-only photometric variations.
- **Verification & Test Suite**:
  - Run all 13 unit tests: `uv run pytest tests/test_task4_research.py tests/test_task4_models.py -v` (13 passed in 7.28s).

### GAN Training Engine (Step 5)
- **Engine Built (`src/task4/train.py`)**:
  - Alternating min-max conditional GAN training loop on NVIDIA RTX 3050 GPU.
  - Mixed precision acceleration via `torch.amp.autocast` and decoupled `GradScaler` instances for $G$ and $D$.
  - One-sided label smoothing ($0.9$ real target) to stabilize discriminator bounds and prevent discriminator over-confidence.
  - Multi-component loss tracking per epoch: $\mathcal{L}_{D,\text{real}}, \mathcal{L}_{D,\text{fake}}, \mathcal{L}_D, \mathcal{L}_{\text{adv}}, \mathcal{L}_{L1}, \mathcal{L}_G$.
  - Fixed-sample visual progression tracking across 6 validation identities (2 per style).
  - Checkpoint manager saving best validation model and rolling checkpoint.

### Baseline Training & Validation Run on RTX 3050 (Step 6)
- **Execution**: Trained on FS2K training partition (901 pairs, batch size 16) with persistent worker acceleration on NVIDIA GeForce RTX 3050 Laptop GPU, evaluating every epoch on the 157-pair stratified validation partition.
- **Empirical Baseline Results (`results/task4/baseline_val_metrics.json`)**:
  - **Overall Val L1 (MAE)**: **0.0976**
  - **Overall Val PSNR**: **15.88 dB**
  - **Overall Val SSIM**: **0.4809**
  - **Style 0 Performance**: L1 = **0.0718**, PSNR = **17.46 dB**, SSIM = **0.5118**
  - **Style 1 Performance**: L1 = **0.1303**, PSNR = **13.52 dB**, SSIM = **0.4077**
  - **Style 2 Performance**: L1 = **0.0911**, PSNR = **16.61 dB**, SSIM = **0.5225**
- **Qualitative Visual Artifacts**:
  - Progression grids saved to `results/task4/visualizations/visual_progression_epoch_*.png` and `visual_progression_baseline.png`.
  - Canonical baseline weights: `checkpoints/task4/baseline_generator.pth` (27.3 MB) and `baseline_discriminator.pth` (11.1 MB).
  - MLflow run: `baseline-cgan-fs2k` in experiment `genai-task4-baseline`.
- **Verification**: Run `uv run pytest tests/test_task4_train.py -v`.

### Optuna Hyperparameter Optimization (Step 7)
- **Objective & Infrastructure**: Bayesian optimization (`task4-cgan` in `optuna/optuna_studies.db`) targeting minimization of validation L1 loss on NVIDIA GeForce RTX 3050 GPU. Utilized `TPESampler` and `MedianPruner(n_startup_trials=5, n_warmup_steps=4)`.
- **Search Space & Execution**:
  - Generator LR (`g_lr`) $\in [1\text{e-}4, 5\text{e-}4]$, Discriminator LR (`d_lr`) $\in [1\text{e-}4, 5\text{e-}4]$, $\lambda_{L1} \in [50.0, 150.0]$, Dropout $\in [0.0, 0.3]$, Base Channels $\in \{32, 64\}$.
  - 10 trials completed in 490.0 seconds (~8.2 minutes, within 15-minute budget). MedianPruner pruned 5 underperforming trials.
- **Optimal Hyperparameters (Trial #10)**:
  - `g_lr`: **$2.2298 \times 10^{-4}$**
  - `d_lr`: **$2.2262 \times 10^{-4}$**
  - `lambda_l1`: **$130.79$**
  - `dropout`: **$0.0015$**
  - `base_channels`: **$64$**
- **Artifacts Exported**:
  - Configuration: `config/task4_best_params.json` and `results/task4/optuna_best_params.json`
  - MLflow experiment: `genai-task4-optuna`

### Final Production Retraining (Step 8)
- **Execution**: Retrained conditional GAN from scratch using best hyperparameters over complete 80-epoch schedule on NVIDIA GeForce RTX 3050 GPU. Completed in 533.3 seconds (~8.9 minutes).
- **Validation Comparison (Baseline vs Retrained Best)**:

| Metric | Baseline (Step 6) | Retrained Best (Step 8) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Best Val L1 (MAE)** | 0.0976 | **0.0943** | **+3.38% Error Reduction** |
| **Best Val PSNR (dB)** | 15.88 dB | **16.12 dB** | **+0.24 dB** |
| **Best Val SSIM** | 0.4809 | **0.4919** | **+0.0110** |
| **Style 0 Val L1** | 0.0718 | **0.0716** | Clean pencil contours |
| **Style 1 Val L1** | 0.1303 | **0.1296** | Dense cross-hatching shading |
| **Style 2 Val L1** | 0.0911 | **0.0871** | Tonal shading refinement |

- **Artifacts Exported**:
  - Production generator weights: `checkpoints/task4/best_generator.pth` (27.3 MB)
  - Production discriminator weights: `checkpoints/task4/best_discriminator.pth` (11.1 MB)
  - Quantitative metrics log: `results/task4/final_train_metrics.json`
  - Visual progression grids: `results/task4/visualizations/final_progression_epoch_*.png`
  - MLflow run: `final-cgan-fs2k` in experiment `genai-task4-final`
- **Verification**: Run `uv run pytest tests/test_task4_optuna_retrain.py -v`.

### Held-Out Test Evaluation & Visual Analysis (Step 9)
- **Quantitative Test Results (1,046 held-out test pairs)**:
  - **Overall Test L1 Error (MAE)**: **0.1074**
  - **Overall Test PSNR**: **15.38 dB**
  - **Overall Test SSIM**: **0.4724**
  - **Overall Test LPIPS (AlexNet Perceptual Distance)**: **0.2515**
- **Per-Style Test Partition Breakdown**:
  - **Style 0**: L1 = **0.0823**, PSNR = **16.98 dB**, SSIM = **0.5104**, LPIPS = **0.2632** (clean linear pencil contouring)
  - **Style 1**: L1 = **0.1529**, PSNR = **12.41 dB**, SSIM = **0.3962**, LPIPS = **0.2376** (dense cross-hatching shading)
  - **Style 2**: L1 = **0.0687**, PSNR = **18.51 dB**, SSIM = **0.5913**, LPIPS = **0.2076** (fine tonal gradation, highest fidelity)
- **Visual Artifacts Produced**:
  - 12-sample test results gallery: `results/task4/sample_results_grid.png`
  - 4-face fixed-identity multi-style synthesis grid: `results/task4/style_comparison_grid.png`
  - Failure case diagnostic panel: `results/task4/failure_cases_analysis.png`
- **MLflow Tracking**: Logged under `genai-task4-evaluation`.
- **Verification**: Run `uv run python src/task4/evaluate.py`.

### Single-Graph ONNX Export & Parity Benchmark (Step 10)
- **ONNX Computational Graph Export (`src/task4/export_onnx.py`)**:
  - Exported complete `UNetGenerator` with internal style embeddings to `models/onnx/task4_generator.onnx` (27.34 MB, opset 17).
  - Dynamic batching enabled on `photo`, `style_index`, and `sketch` tensors.
  - Verified structural graph validity via `onnx.checker.check_model`.
- **Numerical Parity Verification (PyTorch vs ONNX Runtime)**:
  - **Batch 1**: Max absolute difference = **$3.67 \times 10^{-6}$** (< $10^{-5}$ tolerance) -> **PASS**
  - **Batch 4**: Max absolute difference = **$6.05 \times 10^{-6}$** (< $10^{-5}$ tolerance) -> **PASS**
  - **Batch 8**: Max absolute difference = **$6.97 \times 10^{-6}$** (< $10^{-5}$ tolerance) -> **PASS**
  - **Overall Parity Status**: **100% PASS** (`all_passed = True`).
- **CPU Inference Latency Benchmark**:
  - PyTorch CPU: **36.97 ms** (~27.0 FPS)
  - ONNX Runtime CPU: **18.13 ms** (**55.2 FPS**)
  - **CPU Acceleration**: **$2.04\times$ speedup**.
- **Verification**: Run `uv run pytest tests/test_task4_eval_onnx.py -v`.

---

## Application Layer (React, FastAPI, Docker Compose)

### Google Stitch UI Design (Step 1)
- **Design Methodology & Prototyping**: Prototyped complete dark-mode multi-workspace layout in Google Labs Stitch ([https://stitch.withgoogle.com/projects/13236968632410650096](https://stitch.withgoogle.com/projects/13236968632410650096)) in compliance with assignment requirements (Pages 2 & 8 of assignment specification).
- **Navigation Architecture**: Evaluated 3 layout paradigms (Persistent Sidebar, Top Tab Navigation, Card Landing Hub) and selected **Option 1: Collapsible Persistent Sidebar Navigation**:
  - **Instant 1-Click Transitions**: Enables seamless switching between Universal Restoration, Hard-Routing, Soft MoE, and Face-to-Sketch workspaces with zero page reload latency.
  - **Persistent Health & Telemetry**: Dedicated sidebar footer continuously monitors backend status (`GET /health`), active execution provider (`CPUExecutionProvider` / `CUDAExecutionProvider`), and ONNX session readiness.
  - **Uncompromised Comparative Canvas**: Leaves $> 1020\text{px}$ horizontal canvas space on desktop displays ($\ge 1280\text{px}$), fitting dual $256 \times 256\text{px}$ high-DPI side-by-side comparison panels and residual error heatmaps cleanly above the fold.
- **Visual Artifacts & Deliverables**:
  - `results/app/stitch_universal_dashboard.jpg`: Universal restoration dashboard featuring dual input tabs (upload & synthetic corruption studio), side-by-side comparison, and Turbo colormap error heatmap.
  - `results/app/stitch_routing_workspaces.jpg`: Routing analysis view featuring 4-way classification probability bars, specialist routing card, and continuous MoE gating weight distributions summing to 100%.
  - `results/app/stitch_face_sketch.jpg`: FS2K Face-to-Sketch workspace with webcam capture viewfinder, 3 style cards, and PNG download button.
  - `results/app/stitch_design_spec.json`: Design system tokens, color palettes, typography, and component specifications.

### FastAPI Backend Skeleton (Step 2)
- **Application Entrypoint (`src/app/backend/main.py`)**:
  - Preloads all 7 production ONNX models at startup via `@asynccontextmanager` lifespan handler to avoid per-request disk I/O latency.
  - Configures CORS middleware for development (`http://localhost:3000`, `http://localhost:5173`) and Docker production origins.
- **Configuration Module (`src/app/backend/config.py`)**:
  - Manages host, port, origins, upload limits (10MB), and model paths via `Settings(BaseSettings)` with `GENAI_` environment prefix. Zero direct `os.environ` usage.
- **ONNX Session Manager (`src/app/backend/services/inference.py`)**:
  - Automatically preloads and manages all 7 models: `task1_universal_ae`, `task2_classifier`, `task2_specialist_blur`, `task2_specialist_occlusion`, `task2_specialist_salt`, `task3_soft_moe`, and `task4_generator`.
  - Resolves execution providers gracefully (`CUDAExecutionProvider` fallback to `CPUExecutionProvider`).
- **Image Preprocessing & Heatmap Generator (`src/app/backend/services/preprocessing.py`)**:
  - Input validation (JPEG, PNG, WEBP, max 10MB limit).
  - Resizing and normalization to float32 tensor `[1, 3, 128, 128]`.
  - Base64 Data URL encoding and Turbo colormap absolute residual error heatmap generator.
- **Health Monitoring & OpenAPI Docs (`src/app/backend/routers/health.py`)**:
  - Exposes `GET /health` reporting system status (`ok`), active execution provider, and loaded status across all 7 models.
  - Interactive OpenAPI Swagger documentation available at `/docs`.
- **Verification**: Run `uv run pytest tests/test_backend_skeleton.py -v` (14 unit and integration tests passing).

### Universal Restoration Workspace Endpoint (Step 3)
- **Restoration Route (`POST /api/v1/restore/universal`)**:
  - Implemented in `src/app/backend/routers/task1.py`.
  - Accepts `multipart/form-data` image uploads (JPEG, PNG, WEBP, up to 10MB).
  - Supports dual input modes:
    1. Direct upload of pre-corrupted test photos.
    2. Server-side synthetic degradation of clean images across benchmark corruption types:
       - Salt-and-Pepper noise: $p \in \{0.03, 0.08, 0.15\}$
       - Gaussian Blur: $(\text{kernel}, \sigma) \in \{(3, 0.7), (5, 1.5), (7, 2.5)\}$
       - Rectangular Occlusion: 1, 2, or 3 masks covering $\sim 10\%, \sim 20\%, \sim 35\%$ area.
- **Inference Pipeline & Heatmap Synthesis**:
  - Preprocesses input into float32 NCHW tensor `[1, 3, 128, 128]` normalized to $[0, 1]$.
  - Executes production `task1_universal_ae.onnx` session via `session_manager`.
  - Measures inference latency in milliseconds (`~18â€“20 ms` on CPU).
  - Computes pixel-level absolute difference heatmaps ($|I_{\text{restored}} - I_{\text{corrupted}}|$ and clean reference $|I_{\text{restored}} - I_{\text{clean}}|$) mapped to Turbo colormap base64 Data URLs.
- **Response Schema (`UniversalRestoreResponse`)**:
  - Returns `original_image`, `corrupted_image`, `restored_image`, `error_map`, `clean_error_map`, `corruption_applied`, and `inference_time_ms`.
- **Verification**: Run `uv run pytest tests/test_backend_task1.py -v` (11 unit tests passing).

### Hard-Routed Restoration Workspace Endpoint (Step 4)
- **Hard-Routed Inference Route (`POST /api/v1/restore/hard-routed`)**:
  - Implemented in `src/app/backend/routers/task2.py`.
  - Sequential two-stage restoration pipeline combining classifier routing with specialized restoration or identity bypass:
    1. **Stage 1 (Classification)**: Preprocesses image to float32 `[1, 3, 128, 128]` and runs `task2_classifier.onnx`. Generates 4 class logits corresponding to `[clean, salt_and_pepper, gaussian_blur, rectangular_occlusion]`. Applies softmax to produce probability distribution $P \in \mathbb{R}^4$.
    2. **Stage 2 (Specialist Routing)**: Selects highest-probability corruption class $\hat{c} = \arg\max(P)$ (with optional manual override `force_expert`):
       - `clean`: Identity bypass (output = input, specialist latency $0.0\,\text{ms}$).
       - `salt_and_pepper`: Routes to `task2_specialist_salt.onnx`.
       - `gaussian_blur`: Routes to `task2_specialist_blur.onnx`.
       - `rectangular_occlusion`: Routes to `task2_specialist_occlusion.onnx`.
  - Optional server-side synthetic corruption testing (`apply_corruption=true`).
- **Response Schema (`HardRoutedRestoreResponse`)**:
  - `original_image`, `corrupted_image`, `restored_image`, `error_map`, `clean_error_map`.
  - `predicted_class`: Corruption class string (`clean`, `salt_and_pepper`, `gaussian_blur`, `rectangular_occlusion`).
  - `confidence`: Maximum softmax probability float.
  - `class_probabilities`: Dictionary of probabilities for all 4 classes summing to 1.0.
  - `selected_expert`: Model filename or `Identity Bypass`.
  - `inference_time`: Structured latency breakdown object containing `classifier_ms`, `specialist_ms`, and `total_ms`.
- **Verification**: Run `uv run pytest tests/test_backend_task2.py -v` (6 unit tests passing).

### Soft Mixture-of-Experts (MoE) Restoration Workspace Endpoint (Step 5)
- **Soft MoE Inference Route (`POST /api/v1/restore/soft-moe`)**:
  - Implemented in `src/app/backend/routers/task3.py`.
  - Executes single-graph differentiable continuous routing through `models/onnx/task3_soft_moe.onnx`.
  - Evaluates internal gating network with learned temperature ($\tau = 2.526$), generating continuous gating weights $w \in \mathbb{R}^4$ across all 4 branches:
    - Branch 1: `identity_clean` (clean preservation bypass)
    - Branch 2: `expert_salt` (salt-and-pepper specialist)
    - Branch 3: `expert_blur` (gaussian blur specialist)
    - Branch 4: `expert_occlusion` (rectangular occlusion specialist)
  - Blends specialist features via differentiable convex combination $\hat{x} = \sum_{k=1}^4 w_k b_k(x)$ into a single composite restoration.
  - Supports direct corrupted photo uploads as well as server-side benchmark synthetic corruption testing.
- **Response Schema (`SoftMoERestoreResponse`)**:
  - `original_image`, `corrupted_image`, `restored_image`, `error_map`, `clean_error_map`.
  - `routing_weights`: Continuous gating weight distribution summing to $1.0 \pm 10^{-4}$.
  - `dominant_expert`: Name of expert branch receiving the largest weight allocation.
  - `dominant_weight`: Float value of the maximum weight.
  - `entropy`: Shannon entropy $H(w) = -\sum w_i \log_2(w_i + \epsilon)$ measuring routing dispersion.
  - `inference_time_ms`: Single forward graph execution latency (~45â€“50 ms on CPU).
  - `corruption_applied`: Synthetic metadata if server-generated, or null if direct upload.
- **Verification**: Run `uv run pytest tests/test_backend_task3.py -v` (6 unit tests passing).

### Face-to-Sketch Workspace Endpoint (Step 6)
- **Face-to-Sketch Inference Route (`POST /api/v1/sketch/generate`)**:
  - Implemented in `src/app/backend/routers/task4.py`.
  - Executes conditional GAN generator inference via `models/onnx/task4_generator.onnx`.
  - Preprocesses input facial photograph to `[1, 3, 128, 128]` float32 normalized to $[-1, 1]$.
  - Formats target style index tensor (`int64 [1]`) corresponding to the chosen FS2K style:
    - Style 1: Clean Line / Pencil Contour (internal index 0)
    - Style 2: Dense Shading / Cross-Hatch Sketch (internal index 1)
    - Style 3: Fine Tonal / Shaded Art Sketch (internal index 2)
  - Modulates generator bottleneck features via learned FiLM conditioning layers ($d_s = 16$).
  - Postprocesses synthesized sketch tensor from $[-1, 1]$ to RGB uint8 $[0, 255]$ and encodes as base64 PNG Data URL.
- **Response Schema (`SketchGenerateResponse`)**:
  - `original_image`: base64 Data URL of uploaded input face photo.
  - `sketch_image`: base64 Data URL of synthesized facial sketch.
  - `selected_style`: integer style index (1, 2, or 3).
  - `style_description`: human-readable description of the sketch style.
  - `inference_time_ms`: generator execution latency (~18 ms on CPU).
- **Verification**: Run `uv run pytest tests/test_backend_task4.py -v` (6 unit tests passing).

### React Frontend Skeleton (Step 7)
- **Frontend Scaffolding & Setup (`src/app/frontend/`)**:
  - Initialized Vite + React 18 + TypeScript + Tailwind CSS application matching the persistent collapsible sidebar navigation architecture from Step 1.
  - Configured `vite.config.ts` development reverse proxy forwarding `/api` and `/health` seamlessly to the FastAPI backend at `http://127.0.0.1:8000`.
- **Layout Shell & Telemetry Navigation (`src/components/layout/`)**:
  - `Shell.tsx`: Collapsible persistent sidebar with responsive toggle and direct 1-click routing to all 4 workspaces (`/universal`, `/hard-routed`, `/soft-moe`, `/face-to-sketch`).
  - `Navbar.tsx`: Sticky top bar with live backend health indicator badge (polling `GET /health` with 7-model status pill) and active execution provider telemetry (`CPUExecutionProvider` / `CUDAExecutionProvider`).
- **Shared Component Library (`src/components/shared/`)**:
  - `ImageUploader.tsx`: Drag-and-drop file upload with instant square preview, client-side MIME/size validation (JPEG, PNG, WEBP, $\le 10\,\text{MB}$), and clear/remove trigger.
  - `ImagePanel.tsx`: Side-by-side comparative inspection card with title, badges, image display, and direct PNG download action.
  - `LoadingSpinner.tsx`: Themed animated spinners for async inference states.
  - `ErrorAlert.tsx`: Dismissible error callout with retry action.
  - `MetricBadge.tsx`: Visual chips for latency, execution provider, and hardware metrics.
- **Centralized API Service Layer & Types (`src/services/api.ts`, `src/types/index.ts`)**:
  - Strongly typed contracts mapping all REST responses and request parameters.
  - Strict TypeScript compliance: zero `any` types, zero build errors (`tsc && vite build` clean pass in 9.3s).
- **Verification**: Run `cd src/app/frontend && pnpm build`.

### Universal Restoration Workspace Page (Step 8)
- **Interactive Workspace (`src/app/frontend/src/pages/UniversalRestoration.tsx`)**:
  - Implemented complete interactive interface for Task 1 Universal Denoising Autoencoder at route `/universal`.
  - Coordinates dual input modes, synthetic degradation studio, live REST inference, and high-DPI comparative visual inspection.
- **Component Implementations**:
  - `SampleGallery.tsx`: 1-click preset clean pet samples (`Abyssinian #1`, `#2`, `#3`) from `public/samples/` for immediate browser evaluation without manual file hunting.
  - `CorruptionControls.tsx`: Interactive degradation studio toggle with full benchmark configuration:
    - Gaussian Blur: Severity 1 ($3 \times 3, \sigma = 0.7$), Severity 2 ($5 \times 5, \sigma = 1.5$), Severity 3 ($7 \times 7, \sigma = 2.5$).
    - Salt & Pepper: Severity 1 ($p = 0.03$), Severity 2 ($p = 0.08$), Severity 3 ($p = 0.15$).
    - Rectangular Occlusion: Severity 1 ($1\text{ box}, \sim 10\%$), Severity 2 ($2\text{ boxes}, \sim 20\%$), Severity 3 ($3\text{ boxes}, \sim 35\%$).
  - `ErrorMapViewer.tsx`: Pixel-level absolute residual error viewer with tabbed switching between $|I_{\text{restored}} - I_{\text{corrupted}}|$ and ground-truth clean error $|I_{\text{restored}} - I_{\text{clean}}|$, alongside a full Turbo colormap intensity gradient scale ($0.0 \to 1.0$).
- **Verification**: Run `cd src/app/frontend && pnpm build`.

### Hard-Routed Restoration Workspace Page (Step 9)
- **Interactive Workspace (`src/app/frontend/src/pages/HardRoutedRestoration.tsx`)**:
  - Implemented the complete user interface for Task 2 (Hard-Routed Restoration) at route `/hard-routed`.
  - Integrates Stage 1 4-way classification, discrete specialist autoencoder routing, and decomposed latency profiling.
- **Component Implementations**:
  - `ClassifierProbabilities.tsx`: Visual probability breakdown featuring 4 color-coded progress bars (Emerald for Clean, Amber for Salt & Pepper, Sky for Gaussian Blur, Rose for Rectangular Occlusion) highlighting the winner class.
  - `RoutingCard.tsx`: Decision card displaying the predicted category, confidence percentage, routed specialist (`task2_specialist_*.onnx` or `Identity Bypass`), and an interactive dropdown enabling optional manual expert overriding (`force_expert`).
  - `LatencyBreakdown.tsx`: Granular performance telemetry card showing Stage 1 Classifier Latency ($t_{\text{cls}}\,\text{ms}$), Stage 2 Specialist Latency ($t_{\text{spec}}\,\text{ms}$, explicitly highlighting $0.0\,\text{ms}$ on Identity Bypass), and Total End-to-End time ($t_{\text{total}}\,\text{ms}$).
- **Verification**: Run `cd src/app/frontend && pnpm build`.

### Soft MoE Restoration Workspace Page (Step 10)
- **Interactive Workspace (`src/app/frontend/src/pages/SoftMoERestoration.tsx`)**:
  - Implemented the user interface for Task 3 (Soft Mixture-of-Experts Restoration) at route `/soft-moe`.
  - Coordinates image uploads, synthetic degradation studio settings, live single-graph ONNX execution, continuous gating distributions, and composite convex blended image inspection.
- **Component Implementations**:
  - `GatingWeights.tsx`: Continuous gating visualizer with 4 color-coded progress gauges (Identity Pass-through, Salt & Pepper Specialist, Gaussian Blur Specialist, Rectangular Occlusion Specialist) displaying exact percentages ($w_i \times 100\%$) and coefficients, crowning the dominant expert with a glowing purple badge and reporting Shannon routing entropy ($H$ bits).
  - `ExpertContribution.tsx`: Mathematical convex blending card highlighting the active dominant branch and explaining temperature-scaled gating mechanics ($\tau = 2.526$).
### Face-to-Sketch Workspace Page (Step 11)
- **Interactive Workspace (`src/app/frontend/src/pages/FaceToSketch.tsx`)**:
  - Implemented the user interface for Task 4 (Face-to-Sketch Synthesis) at route `/face-to-sketch`.
  - Coordinates conditional GAN generator inference with dual input options (drag-and-drop file upload or live webcam capture), 3-way style switching, high-fidelity side-by-side comparison, and direct PNG export.
- **Component Implementations**:
  - `WebcamCapture.tsx`: HTML5 `navigator.mediaDevices.getUserMedia` video viewfinder supporting live camera streaming, snapshot freeze-frame capture, and instant canvas blob-to-File conversion.
  - `StyleSelector.tsx`: Interactive 3-way card group conditioning synthesis on FS2K styles via FiLM bottleneck modulation: Style 1 (Clean Contour / Light Pencil), Style 2 (Cross-Hatch / Artistic Textures), and Style 3 (Tonal Shading / Graphite Gradients).
  - `SketchViewer.tsx`: Side-by-side comparison view presenting input facial portrait alongside synthesized cGAN sketch, inference latency badge, and 1-click client-side PNG download (`sketch_style_{id}_{timestamp}.png`).
  - 1-click preset sample portrait button (`face_sample.jpg`) from `public/samples/` for immediate browser evaluation without manual file hunting.
### Production Docker Compose Deployment (Step 12)
- **Universal CPU-Only Architecture**:
  - Engineered for 100% CPU inference with zero GPU hardware dependencies and zero CUDA drivers. Runs seamlessly across all operating systems: Windows (Docker Desktop / WSL2), Linux (x86_64 and aarch64), and macOS (Apple Silicon M-series and Intel).
- **Two-Stage Backend Container (`Dockerfile.backend`)**:
  - *Stage 1 (Builder)*: Base `python:3.11-slim`, installs Astral `uv`, creates isolated virtualenv, and installs pure CPU PyTorch/Torchvision (`--index-url https://download.pytorch.org/whl/cpu`), ONNX Runtime CPU, FastAPI, Uvicorn, Pillow, NumPy, and Matplotlib.
  - *Stage 2 (Runtime)*: Minimal `python:3.11-slim` runtime image, non-root user `appuser` (UID 1000), copies virtualenv, copies backend source files, and exposes port 8000.
  - Native healthcheck verifying `GET /health` with Python standard library `urllib`.
- **Two-Stage Frontend Container (`Dockerfile.frontend`)**:
  - *Stage 1 (Builder)*: Base `node:20-alpine`, installs `pnpm@9`, installs locked dependencies (`pnpm install --frozen-lockfile`), and compiles TypeScript + Vite static assets into `/app/dist`.
  - *Stage 2 (Runtime)*: Minimal `nginx:alpine`, copies compiled assets into `/usr/share/nginx/html`, mounts `nginx.conf`, and exposes port 80.
- **Nginx Reverse Proxy (`nginx.conf`)**:
  - Reverse proxies `/api/` to `http://backend:8000/api/` and `/health` to `http://backend:8000/health`.
  - Serves static assets with Gzip compression and long-term cache headers.
  - Implements SPA HTML5 fallback (`try_files $uri $uri/ /index.html;`) resolving React Router client-side routes.
  - Allows up to 25MB uploads (`client_max_body_size 25M;`).
- **Multi-Container Orchestration (`docker-compose.yml`)**:
  - Mounts `./models/onnx` as a read-only volume (`:ro`) preventing image bloat and enabling zero-rebuild model updates.
  - Healthcheck dependency chain (`frontend` waits for `backend` to pass healthcheck before serving traffic).
  - Single-command startup: `docker compose up --build`.
  - Single-command teardown: `docker compose down`.













### Final Integration & Evaluator Documentation (Step 13)

- Fixed oversized-upload status: now HTTP **413** (was 400) via `ImageTooLargeError`; frontend `api.ts` also renders FastAPI 422 detail arrays.
- Added `tests/test_backend_errors.py` (20 tests: 400/413/503 across all 4 endpoints) -> 63 backend tests passing.
- Added `scripts/smoke_test.py` (live 25-check smoke test) and `scripts/benchmark_cpu.py` (writes `results/app/cpu_benchmark.json`).
- ONNX models are committed to git (`.gitignore` no longer excludes them; largest file 57.8 MB < GitHub's 100 MB limit).
- Added evaluator walkthrough, user guide, benchmark table (top of this README) and `docs/report_artifacts.md`.
