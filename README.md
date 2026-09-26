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

### 4. Verify Setup
```bash
task --list
```
