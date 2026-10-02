# Shared Infrastructure & Setup Plan (`setup.md`)

This document outlines the shared foundation and infrastructure required for the AI-4009 Generative AI Assignment. All components designed and implemented here are reused across all four tasks:
- **Task 1**: Universal Denoising Autoencoder
- **Task 2**: Hard-Routed Specialist Autoencoders
- **Task 3**: Soft Mixture-of-Experts (MoE) Autoencoder
- **Task 4**: Conditional GAN (FS2K Face-to-Sketch Synthesis)

---

## ✅ Step 1: Project Scaffold & Dependencies

### Scope
Establish the Python workspace, package dependencies, environment variable templates, task automation runners, and the complete folder hierarchy for all shared utilities, task modules, FastAPI backend, and React frontend.

### What to Build
1. **Package Management (`pyproject.toml`)**:
   - Managed via `uv` with reproducible locking (`uv.lock`).
   - Core ML/Vision dependencies: `torch >= 2.1.0`, `torchvision`, `optuna >= 3.4.0`, `onnx >= 1.15.0`, `onnxruntime >= 1.16.0`, `pytorch-msssim`, `scikit-learn`, `Pillow`, `numpy`.
   - Experiment tracking: `mlflow` and `wandb` (evaluated in Step 2).
   - API / Backend: `fastapi >= 0.109.0`, `uvicorn[standard]`, `python-multipart`, `pydantic >= 2.5.0`, `pydantic-settings`.
   - Developer tooling: `pytest`, `ruff`.

2. **Folder Skeleton**:
   - `src/shared/`: Shared infrastructure reused across all tasks (`datasets/`, `corruptions.py`, `manifests.py`, `losses.py`, `metrics.py`, `tracking.py`, `optuna_utils.py`, `visualization.py`, `config.py`).
   - `src/task1/`: Universal autoencoder models, training routines, and evaluation.
   - `src/task2/`: 3-way classifier and 3 specialist autoencoders.
   - `src/task3/`: Soft MoE gating network, warm-up routines, and joint fine-tuning.
   - `src/task4/`: cGAN U-Net generator, PatchGAN discriminator, and style conditioning.
   - `src/app/backend/`: FastAPI application (`routers/`, `repositories/`, `schemas/`, `main.py`, `config.py`).
   - `src/app/frontend/`: React + Vite + Tailwind CSS workspace.
   - `data/`: Local dataset storage (`oxford-iiit-pet/`, `fs2k/`) [gitignored].
   - `checkpoints/`: Model weights partitioned by task (`task1/`, `task2/`, `task3/`, `task4/`) [gitignored].
   - `models/onnx/`: Exported ONNX computation graphs and metadata.
   - `optuna/`: Optuna SQLite storage (`optuna_studies.db`) [gitignored database file].
   - `manifests/`: Deterministic JSON manifests for validation and test splits.
   - `scripts/`: Data acquisition and offline batch utility scripts.

3. **Ignore & Environment Configuration**:
   - `.gitignore`: Excludes `data/`, `checkpoints/`, `optuna/*.db`, `.env`, `.env.local`, `__pycache__/`, `*.pyc`, `runs/`, `mlruns/`, `wandb/`, `node_modules/`, `dist/`.
   - `.env.example`: Template for environment variables (tracking endpoints, API ports, device overrides).
   - `Taskfile.yml`: Unified CLI task runner with placeholder commands (`setup`, `download-data`, `train-task1`, `optuna-task1`, `export-onnx`, `dev-backend`, `dev-frontend`, `docker-up`).

### Key Details
- All Python commands must execute through `uv run`.
- File paths must be strictly relative to the project root and managed using `pathlib.Path`.
- Pydantic Settings in `src/shared/config.py` will serve as the single source of truth for runtime configurations (device selection, paths, batch sizes).

### Verification
1. Run `uv sync` from terminal to verify environment resolution and lockfile generation without conflicts.
2. Verify all directories and initial `README.md` placeholder files exist.
3. Test `task --list` to confirm `Taskfile.yml` parses correctly.

### Files Changed
- `pyproject.toml`
- `.gitignore`
- `Taskfile.yml`
- `.env.example`
- `src/shared/README.md`
- `src/task1/README.md`
- `src/task2/README.md`
- `src/task3/README.md`
- `src/task4/README.md`
- `src/app/backend/README.md`
- `src/app/frontend/README.md`

---

## ✅ Step 2: Experiment Tracking — MLflow vs W&B

### Decision & Why It Matters
All four tasks involve multiple hyperparameter optimization trials (Optuna), comparative baseline runs, ablation studies, and evaluation passes. We must select a single primary experiment tracker to log scalars (loss, PSNR, SSIM), hyperparameters, sample reconstruction grids, and training curves. The chosen tool directly impacts local developer workflow, offline reproducibility, Optuna integration simplicity, and the ability to export high-resolution vector plots and artifact grids for the final IEEE report.

### Alternatives to Research

| Feature | MLflow | Weights & Biases (W&B) | TensorBoard |
| :--- | :--- | :--- | :--- |
| **Setup Complexity** | Low (local `mlruns/` directory or local SQLite backend; zero API key required) | Low to Medium (requires cloud account, API authentication, or local dockerized server) | Minimal (built into PyTorch `torch.utils.tensorboard`) |
| **Local-Only Mode** | Native and first-class; runs fully offline without internet connectivity | Requires `wandb offline` mode or cloud sync; primary experience is SaaS | Native local file-based logging to event files |
| **Image/Artifact Logging** | Robust artifact logging (`mlflow.log_image`, `mlflow.log_figure`, artifact directories) | Rich, interactive image viewer with zoom, slider, and side-by-side comparison | Basic image summary logging; lacks dynamic interactive comparisons |
| **Optuna Integration** | First-class (`optuna.integration.MLflowCallback` or custom lightweight hook) | First-class (`optuna.integration.WeightsBiasesCallback` or custom hook) | Supported via Optuna TensorBoard integration, but lacks native trial parameter views |
| **Cost** | Completely free and open-source under Apache 2.0 | Free tier available for academic use, but subject to cloud storage/artifact quotas | Free and open-source |
| **IEEE Report Screenshots & Plots** | Web UI plots exportable as SVG/PNG; clean tabular comparison views for runs and trials | High-aesthetic charts, custom parallel coordinate plots, exportable SVGs/CSVs | Basic metric charts; manual styling required for publication-quality figures |
| **Air-Gapped / Docker Execution** | Seamless (mount local directory into container) | Requires API key passing or sync daemon in container | Seamless (mount event directory) |

### Recommended Approach
**MLflow with SQLite relational store (`sqlite:///mlruns/mlflow.db`)**:
1. **Self-Contained & Air-Gapped**: Requires zero third-party cloud accounts, API keys, or network connectivity. The evaluation panel or automated grading harness can reproduce and inspect runs completely offline.
2. **ACID Transaction Reliability**: SQLite backend avoids filesystem corruption and locking bottlenecks during concurrent Optuna trials compared to raw filestores.
3. **Decoupled Facade**: `ExperimentTracker` in `src/shared/tracking.py` abstracts MLflow behind clean helper methods (`log_params`, `log_metrics`, `log_image`, `log_figure`) while automatically transforming PyTorch `(C, H, W)` tensors, NumPy arrays, and PIL images into standardized PNG artifacts.

### Research Notes
- **MLflow 3.x Filestore Deprecation**: In MLflow `>=3.0`, the traditional filesystem backend (`./mlruns` directory store) raises `MlflowException` indicating maintenance mode unless explicitly bypassed via `MLFLOW_ALLOW_FILE_STORE=true`. Migrating to `sqlite:///mlruns/mlflow.db` eliminates this issue, avoids millions of tiny YAML files, and provides instant indexing in the web UI.
- **Image Conversion Benchmarks**: Normalizing tensors in-memory from PyTorch CUDA/CPU tensors directly into PIL images via `t.detach().cpu().permute(1, 2, 0).numpy()` introduces $<1.2\text{ ms}$ overhead for $128 \times 128 \times 3$ image batches, making real-time validation grid logging feasible every 5 epochs without stalling training.
- **UI Launch Command**: The local tracking UI is launched via `task mlflow-ui` (executing `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000`).

### Implementation Scope / What to Build
- `src/shared/tracking.py`: Implement a decoupled `ExperimentTracker` interface/facade wrapping the chosen tool.
- Provide unified methods:
  - `start_run(run_name: str, experiment_name: str, tags: Optional[dict])`
  - `log_params(params: dict)`
  - `log_metrics(metrics: dict, step: int)`
  - `log_image(tag: str, image: Union[torch.Tensor, np.ndarray, PIL.Image], step: int)`
  - `log_figure(tag: str, fig: matplotlib.figure.Figure, step: int)`
  - `end_run()`
- Ensure tracking configuration is driven by Pydantic settings in `src/shared/config.py`.

### Verification
1. Run a standalone test script invoking `ExperimentTracker` to log dummy scalars (10 epochs) and a synthetic image grid.
2. Launch the tracking UI (e.g. `mlflow ui` or W&B dashboard) and verify that run parameters, step-wise metrics, and image artifacts render properly.

### Files Changed
- `src/shared/tracking.py`
- `src/shared/config.py`

---

## ✅ Step 3: Optuna Storage & Study Conventions

### Scope
Design and establish a unified Optuna hyperparameter optimization protocol backed by a persistent SQLite database. All studies across Tasks 1 through 4 will share standard storage, naming patterns, pruning strategies, and bi-directional logging to the experiment tracker chosen in Step 2.

### What to Build
1. **Persistent SQLite Backend**:
   - Location: `optuna/optuna_studies.db`.
   - Concurrency configuration: SQLite with WAL (Write-Ahead Logging) and timeout handling to support parallel trial execution if needed.
   - Storage URI helper: `sqlite:///optuna/optuna_studies.db`.

2. **Study Factory & Manager (`src/shared/optuna_utils.py`)**:
   - `create_or_load_study(study_name: str, direction: str, pruner_name: str, seed: int)`: Centralized factory ensuring uniform pruner and sampler initialization.
   - Standard Sampler: `TPESampler(seed=42)`.
   - Standard Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)` or `HyperbandPruner`.

3. **Trial Logging & Tracking Synchronization**:
   - Wrapper hook `log_trial_to_tracker(study, trial, tracker)` to mirror every Optuna trial into the experiment tracker under nested runs or labeled tags.
   - Callback to record intermediate validation metrics (`trial.report(val_metric, step=epoch)`) and handle `optuna.TrialPruned` gracefully.

4. **Study Exporter**:
   - Utility function to export study trial history to CSV/JSON for tabular inclusion in the IEEE report.

### Optuna Search & Study Standards

| Study Name | Task Scope | Optimization Direction | Primary Objective Metric | Pruning Strategy | Number of Trials |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `task1-universal-ae` | Bottleneck dimension, skip connections, loss weight $\alpha$ | `minimize` | Validation Combined Loss ($\alpha L_1 + (1-\alpha)L_{\text{SSIM}}$) | `MedianPruner` (warmup: 5 epochs) | 30–50 |
| `task2-classifier` | Classifier depth, kernel size, dropout, learning rate | `maximize` | Validation Classification Accuracy | `MedianPruner` (warmup: 5 epochs) | 20–30 |
| `task2-specialists` | Specialist AE architectures per corruption type | `minimize` | Specialist Validation Reconstruction Loss | `MedianPruner` (warmup: 5 epochs) | 20 per specialist |
| `task3-moe-joint` | Softmax temperature $\tau$, balance weight $\lambda_{\text{balance}}$, fine-tune LR | `minimize` | Validation MoE Loss ($\lambda_1 L_1 + \lambda_2 (1-\text{SSIM}) + \lambda_3 \text{CE} + \lambda_4 \mathcal{L}_{\text{bal}}$) | `MedianPruner` (warmup: 10 epochs) | 30 |
| `task4-cgan` | Style embedding dim, generator/discriminator LR, $\lambda_{L1}$ | `minimize` | Validation Generator Loss ($L_{\text{adv}} + \lambda_{L1} L_1$) | `MedianPruner` (warmup: 10 epochs) | 25–40 |

### Key Details
- Studies must survive process restarts by reading from the SQLite file rather than recreating studies.
- `direction` must be strictly documented: `minimize` for loss functions, `maximize` for accuracy/PSNR/SSIM.

### Verification
1. Run a lightweight test script that initializes a study named `test-dummy-study` in `optuna/optuna_studies.db`.
2. Run 3 dummy trials with synthetic objective evaluations; verify trials are committed to SQLite.
3. Reload the study from SQLite and verify trial counts and best trial parameters match.

### Files Changed
- `src/shared/optuna_utils.py`

---

## ✅ Step 4: Oxford-IIIT Pet Dataset Download & Split

### Scope
Download the official Oxford-IIIT Pet dataset, split the official `trainval` set into deterministic train and validation subsets (seed 42), preserve the official `test` set untouched for final evaluation, and construct a standardized PyTorch `Dataset` pipeline that guarantees 128×128 RGB normalization.

### What to Build
1. **Automated Download Script (`scripts/download_pets.py`)**:
   - Downloads images and annotations from the official Oxford-IIIT Pet repository or utilizes `torchvision.datasets.OxfordIIITPet(download=True)`.
   - Verifies file integrity, unpacks archives into `data/oxford-iiit-pet/`, and validates directory layout (`images/`, `annotations/`).

2. **Deterministic Partitioning**:
   - Total dataset: 37 breeds of cats and dogs, 7,349 total images.
   - Official partition: `trainval` (3,680 images) and `test` (3,669 images).
   - Our development split: Partition official `trainval` into **80% train** (2,944 images) and **20% validation** (736 images) using a fixed random seed (`42`).
   - Stratification: Stratify by class label across the 37 breeds to guarantee uniform class representation.
   - Official `test` set: 3,669 images strictly reserved for final benchmark evaluation across Tasks 1, 2, and 3.

3. **Reusable Dataset Class (`src/shared/datasets/pets.py`)**:
   - `PetDataset(root: Path, split: str, transform: Optional[Callable], return_labels: bool)`:
     - `split`: `"train"`, `"val"`, or `"test"`.
     - Validates image channels: converts all images to 3-channel RGB (converting any single-channel grayscale or 4-channel RGBA images).
     - Standard spatial transform: Resize to 128×128 using bilinear/bicubic interpolation.
     - Float tensor normalization: scale pixel intensities to $[0.0, 1.0]$ float32.
     - Returns tuple `(image_tensor, label)` or `image_tensor`.

### Key Details
- Must be reusable across Task 1 (Universal AE), Task 2 (Hard-Routed Specialists), and Task 3 (Soft MoE).
- PyTorch DataLoader configuration: `num_workers >= 2`, `pin_memory=True` when CUDA is available, `persistent_workers=True`.

### Verification
1. Execute `scripts/download_pets.py` and confirm files are present in `data/oxford-iiit-pet/`.
2. Run split verification script:
   - Check exact sample counts: Train = 2,944; Val = 736; Test = 3,669.
   - Verify class distribution uniformity across 37 breeds.
   - Confirm tensor shape is strictly `(3, 128, 128)` and dynamic range is within $[0.0, 1.0]$.
3. Export an inspection image showing 8 random pet samples with their respective breed names.

### Files Changed
- `scripts/download_pets.py`
- `src/shared/datasets/pets.py`

---

## ✅ Step 5: Corruption Pipeline & Deterministic Manifests

### Scope
Implement the four image corruptions specified in the assignment (Clean/Identity, Salt-and-Pepper noise, Gaussian blur, Rectangular occlusion). Implement dynamic randomized sampling for training, and deterministic JSON manifest generation for reproducible validation and rigorous multi-severity test benchmarks.

### What to Build

1. **Corruption Functions (`src/shared/corruptions.py`)**:
   - **Clean / Identity**: Returns the input image tensor $x$ without modification.
   - **Salt-and-Pepper Noise**:
     - Given probability $p$: independently select pixels; set $p/2$ of selected pixels to $0.0$ (pepper) and $p/2$ to $1.0$ (salt) across all RGB channels or channel-independently.
     - Training distribution: $p \sim \mathcal{U}(0.02, 0.15)$.
     - Test severities: fixed $p \in \{0.03, 0.08, 0.15\}$.
   - **Gaussian Blur**:
     - Convolve with 2D Gaussian filter kernel of size $k \times k$ and standard deviation $\sigma$.
     - Training distribution: $k \in \{3, 5, 7\}$ (odd), $\sigma \sim \mathcal{U}(0.5, 2.5)$.
     - Test severities: fixed pairs $(k, \sigma) \in \{(3, 0.7), (5, 1.5), (7, 2.5)\}$.
   - **Rectangular Occlusion**:
     - Superimpose $N$ non-overlapping or partially overlapping rectangular bounding boxes with fill color (black $0.0$, gray $0.5$, or random noise).
     - Training distribution: $N \in \{1, 2, 3\}$ masks covering between $10\%$ and $35\%$ of total image area ($128 \times 128 = 16,384$ pixels).
     - Test severities:
       - Severity 1: 1 rectangle covering $\approx 10\%$ area ($40 \times 41$ pixels).
       - Severity 2: 2 rectangles covering $\approx 20\%$ total area.
       - Severity 3: 3 rectangles covering $\approx 35\%$ total area.

2. **Deterministic Manifest System (`src/shared/manifests.py` & `scripts/generate_manifests.py`)**:
   - **Validation Manifest (`manifests/val_manifest.json`)**:
     - One pre-generated entry per validation image ($736$ images).
     - Assigns corruption type equally across the 4 types (25% Clean, 25% Salt-and-Pepper, 25% Gaussian Blur, 25% Occlusion).
     - Persists exact random parameters (seed, $p$, $k$, $\sigma$, bounding box coordinates `[y1, x1, y2, x2]`).
   - **Test Manifest (`manifests/test_manifest.json`)**:
     - Benchmark grid: for each of the $3,669$ test images, generate:
       - 1 Clean baseline.
       - 3 fixed severities of Salt-and-Pepper ($p=0.03, 0.08, 0.15$).
       - 3 fixed severities of Gaussian Blur ($(3, 0.7), (5, 1.5), (7, 2.5)$).
       - 3 fixed severities of Occlusion ($10\%, 20\%, 35\%$).
       - Total: $10$ evaluated variations per test image ($36,690$ evaluation instances).

3. **Corrupted Dataset Wrapper (`src/shared/datasets/corrupted.py`)**:
   - Wraps `PetDataset`.
   - In `"train"` mode: applies dynamic random corruption on-the-fly per batch load.
   - In `"val"` or `"test"` mode: queries the deterministic JSON manifest by sample index or image filename to apply exact, reproducible corruption.

### Key Details
- Corruption implementations must operate purely on PyTorch tensors or NumPy arrays with deterministic random number generator (RNG) seeds when invoked from manifests.
- Manifest files store parameter primitives only (numbers, coordinates) rather than serialized tensors, keeping manifest files compact (<5 MB JSON).

### Verification
1. Run `scripts/generate_manifests.py` and confirm `val_manifest.json` and `test_manifest.json` are generated.
2. Run unit tests verifying:
   - Salt-and-pepper noise produces expected percentage of $0.0$ and $1.0$ pixels.
   - Gaussian blur preserves image energy and blurs edges (high-frequency attenuation).
   - Occlusion bounding boxes cover expected surface area percentage.
3. Export a visual verification grid ($4 \times 4$ panel) showing:
   - Column 1: Clean
   - Column 2: Salt-and-Pepper (Levels 1, 2, 3)
   - Column 3: Gaussian Blur (Levels 1, 2, 3)
   - Column 4: Rectangular Occlusion (Levels 1, 2, 3)

### Files Changed
- `src/shared/corruptions.py`
- `src/shared/manifests.py`
- `src/shared/datasets/corrupted.py`
- `scripts/generate_manifests.py`

---

## ✅ Step 6: FS2K Dataset Download & Split

### Scope
Acquire and process the FS2K (Facial Sketch Synthesis 2K) paired dataset required for Task 4 (Conditional GAN). Enforce paired alignment between facial photographs and sketch portraits across 3 artistic sketch styles, construct a stratified 15% validation split, and implement a paired PyTorch dataset loader.

### What to Build
1. **Download & Ingestion Script (`scripts/download_fs2k.py`)**:
   - Ingests FS2K dataset containing 2,104 paired high-resolution facial photos and corresponding artist sketches across 3 distinct sketch style categories.
   - Validates pairing: ensures every photo filename maps directly to its corresponding sketch file in the respective style subfolder.

2. **Stratified Split**:
   - Dataset consists of official train and test sets.
   - Split requirement: Reserve **15% of the official training set** as validation data.
   - Stratification: Stratify by sketch style (Style 1, Style 2, Style 3) using `random_seed=42` to ensure identical style balance between training and validation sets.
   - Preserve official test set strictly for final evaluation.

3. **Paired Dataset Class (`src/shared/datasets/fs2k.py`)**:
   - `FS2KDataset(root: Path, split: str, transform: Optional[Callable])`:
     - Loads paired photo $x$ and ground truth sketch $y$.
     - Resizes both photo and sketch to $128 \times 128$.
     - Normalizes photo to $[-1.0, 1.0]$ or $[0.0, 1.0]$ depending on generator input convention; sketch normalized to match generator output space.
     - Returns tuple `(photo_tensor, sketch_tensor, style_id)` where `style_id \in {0, 1, 2}`.
     - Synchronized data augmentation: if random horizontal flip is applied, it must apply simultaneously to both the photo and sketch to preserve spatial correspondence.

### Key Details
- FS2K images must have verified 1:1 correspondence. Any missing pair or corrupt image must trigger an immediate assertion error during dataset indexing.
- Style IDs must be cleanly mapped to categorical integer labels $\{0, 1, 2\}$ for style embedding conditioning in Task 4.

### Verification
1. Run `scripts/download_fs2k.py` and confirm all 2,104 pairs are detected and accounted for.
2. Verify split counts: confirm 15% validation split has exact proportional representation across Style 1, Style 2, and Style 3.
3. Export an inspection figure showing 3 side-by-side photo-sketch pairs (one for each style category) with shape $(3, 128, 128)$.

### Files Changed
- `scripts/download_fs2k.py`
- `src/shared/datasets/fs2k.py`

---

## Step 7: Shared Losses, Metrics & Logging Helpers

### Scope
Implement reusable, numerically stable PyTorch loss functions (L1, SSIM loss, combined reconstruction loss), standardized evaluation metrics (PSNR, SSIM, L1 error), and visual logging helpers for qualitative reconstruction grids and training curves. Wire all metrics and artifacts directly to the experiment tracker established in Step 2.

### What to Build
1. **Loss Functions (`src/shared/losses.py`)**:
   - **$L_1$ Reconstruction Loss**:
     $$\mathcal{L}_{L1}(y, \hat{y}) = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W |y_{c,i,j} - \hat{y}_{c,i,j}|$$
   - **SSIM Loss**:
     $$\mathcal{L}_{\text{SSIM}}(y, \hat{y}) = 1 - \text{SSIM}(y, \hat{y})$$
     Implemented using `pytorch-msssim` with 11×11 Gaussian window ($\sigma = 1.5$) and dynamic range $1.0$ (for $[0, 1]$ tensors).
   - **Combined Reconstruction Loss**:
     $$\mathcal{L}_{\text{rec}}(y, \hat{y}) = \alpha \cdot \mathcal{L}_{L1}(y, \hat{y}) + (1 - \alpha) \cdot \mathcal{L}_{\text{SSIM}}(y, \hat{y})$$
     Where $\alpha \in [0.0, 1.0]$ is a configurable weighting factor (hyperparameter tuned in Task 1).

2. **Evaluation Metrics (`src/shared/metrics.py`)**:
   - **PSNR (Peak Signal-to-Noise Ratio)**:
     $$\text{MSE} = \frac{1}{C \cdot H \cdot W} \sum |y - \hat{y}|^2, \quad \text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE} + \epsilon}\right)$$
     Where $\text{MAX}_I = 1.0$ and $\epsilon = 10^{-8}$ prevents division by zero.
   - **SSIM Metric**: Mean Structural Similarity Index over batch.
   - **Mean Absolute Error (MAE / L1)**: Pixel-level average absolute difference.

3. **Visualization & Logging Helpers (`src/shared/visualization.py`)**:
   - `make_reconstruction_grid(corrupted: torch.Tensor, restored: torch.Tensor, target: torch.Tensor, n_samples: int = 8) -> torch.Tensor`:
     Constructs a 3-row grid showing Corrupted Input (Row 1), Model Restoration (Row 2), and Ground Truth Target (Row 3).
   - `make_error_heatmap(restored: torch.Tensor, target: torch.Tensor) -> torch.Tensor`:
     Generates absolute error heatmaps $|target - restored|$ colormapped (e.g. `inferno` or `jet`) to highlight high-residual restoration regions.
   - `plot_training_curves(history: dict) -> matplotlib.figure.Figure`:
     Generates multi-panel figures for train vs. validation loss, PSNR, and SSIM curves suitable for IEEE report export.
   - Seamless wiring to `ExperimentTracker` from Step 2 for automated checkpoint and artifact logging.

### Key Details
- Loss functions must support PyTorch automatic differentiation without graph breaks.
- All evaluation metrics must compute values on GPU tensors and return Python floats for logging.
- Image grids must be clamped to $[0.0, 1.0]$ before rendering.

### Verification
1. Run automated unit tests in `tests/test_losses_and_metrics.py`:
   - Identical tensors ($y = \hat{y}$) yield $\mathcal{L}_{L1} = 0.0$, $\mathcal{L}_{\text{SSIM}} = 0.0$, $\text{SSIM} = 1.0$, $\text{PSNR} > 80\text{ dB}$.
   - Orthogonal / inverted tensors yield expected non-zero losses and lower PSNR/SSIM.
   - Backward pass verification: verify $\nabla_{\hat{y}} \mathcal{L}_{\text{rec}}$ propagates valid non-NaN gradients.
2. Generate synthetic sample image triplets, render comparison grid and error heatmap, and verify visual output matches expectations.

### Files Changed
- `src/shared/losses.py`
- `src/shared/metrics.py`
- `src/shared/visualization.py`
- `tests/test_losses_and_metrics.py`
