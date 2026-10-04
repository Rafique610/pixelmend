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
- **Numerical Parity Verification**: Strict assertion ($\max |Y_{\text{pt}} - Y_{\text{ort}}| < 10^{-5}$) passed across all 4 models for $B \in \{1, 4, 8\}$ (classifier max diff: $5.72 \times 10^{-6}$, specialists max diffs: $2.38 \times 10^{-7}$ – $7.75 \times 10^{-7}$).
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
- **Pruner**: MedianPruner with custom routing collapse guard ($\max \bar{w}_k > 0.90$ or $\min \bar{w}_k < 0.02$). Pruned 12 non-competitive / uncalibrated trials.
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



