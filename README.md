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

