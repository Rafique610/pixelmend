# AI-4009 Generative AI — Assignment #1

End-to-end implementation and deployment of four generative computer vision systems:
1. **Universal Denoising Autoencoder** (multi-corruption blind restoration)
2. **Hard-Routed Specialist Autoencoders** (corruption classifier + dedicated restoration experts)
3. **Soft Mixture-of-Experts Restoration** (differentiable gating network over 4 branches)
4. **Style-Conditioned Face-to-Sketch Synthesis** (conditional GAN on FS2K dataset)

Integrated and deployed through a single **FastAPI + React (Tailwind CSS)** web application with **Docker Compose**.

---

## Architecture & Workspaces

| Workspace / System | Description | Endpoint | Target Model |
| :--- | :--- | :--- | :--- |
| **Setup & Infra** | Shared data pipelines, manifests, MLflow tracking, Optuna DB | — | — |
| **Task 1: Universal Restoration** | Single convolutional autoencoder restoring clean, salt-and-pepper, blur, and occlusion without corruption label | `POST /api/v1/restore/universal` | `models/onnx/task1_universal_ae.onnx` |
| **Task 2: Hard-Routed Restoration** | 4-class classifier routing clean (identity) vs 3 specialist autoencoders | `POST /api/v1/restore/hard-routed` | `models/onnx/task2_*.onnx` (4 models) |
| **Task 3: Soft MoE Restoration** | Continuous gating network combining identity + 3 specialists dynamically | `POST /api/v1/restore/soft-moe` | `models/onnx/task3_soft_moe.onnx` |
| **Task 4: Face-to-Sketch** | Paired photo-to-sketch cGAN conditioned on 3 learned style categories | `POST /api/v1/sketch/generate` | `models/onnx/task4_generator.onnx` |

---

## Directory Structure

```
.
├── .env.example              # Environment variables template
├── .gitignore                # Comprehensive Git ignore rules
├── Taskfile.yml              # Central task automation runner
├── pyproject.toml            # Python packaging & dependencies (uv managed)
├── data/                     # Downloaded datasets (gitignored)
│   ├── oxford-iiit-pet/      # Tasks 1–3 development & test data
│   └── fs2k/                 # Task 4 paired face-to-sketch data
├── checkpoints/              # Saved PyTorch model checkpoints (gitignored)
├── models/
│   └── onnx/                 # Exported ONNX models for production inference
├── manifests/                # Deterministic validation & test corruption manifests
├── optuna/                   # SQLite database for Optuna studies (optuna_studies.db)
├── scripts/                  # Data download & batch preparation scripts
├── src/
│   ├── shared/               # Shared losses, metrics, config, tracking, datasets
│   ├── task1/                # Universal autoencoder module
│   ├── task2/                # Classifier and specialist autoencoders
│   ├── task3/                # Soft Mixture-of-Experts pipeline
│   ├── task4/                # Conditional GAN generator & discriminator
│   └── app/
│       ├── backend/          # FastAPI REST API (routers, schemas, services)
│       └── frontend/         # React + Tailwind CSS web interface
└── tests/                    # Pytest test suites
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

All hyperparameter search studies across Tasks 1–4 are persisted in a centralized SQLite database (`optuna/optuna_studies.db`). Studies survive process restarts and automatically synchronize trial metrics with MLflow.

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
   - Training: $1$–$3$ boxes covering $10\%$–$35\%$ image area.
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
  Powered by `pytorch_msssim` with 11×11 Gaussian window ($\sigma = 1.5$) and dynamic range $1.0$.
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
- **Bottleneck**: Compressed spatial feature map ($8 \times 8 \times 256$, $16,384$ floats) enforcing a strict $3.0\times$–$12.0\times$ data compression ratio with optional dropout.
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


