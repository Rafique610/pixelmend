# Knowledge Domain Map

## Project

| Key | Value |
|-----|-------|
| Name | AI-4009 Generative AI Assignment #1 |
| Domain | Computer Vision — image restoration autoencoders, mixture-of-experts, conditional GANs |
| Deliverables | 4 trained models + unified React/FastAPI app + IEEE report + demo video |

## Stacks in Play

| Stack | Invariant Doc | Notes |
|-------|---------------|-------|
| Universal (all tasks) | `docs/invariants/core.md` | Folder layout, file-size limits, Ponytail ladder |
| ML / Training (PyTorch) | `docs/invariants/ml.md` | PyTorch, Optuna, MLflow/W&B, ONNX, training |
| Backend (FastAPI) | `docs/invariants/backend.md` | FastAPI serving ONNX models |
| Frontend (React) | `docs/invariants/frontend.md` | React + Tailwind CSS |
| Security | `docs/invariants/security.md` | Credentials, env vars |
| Tooling | `docs/invariants/tooling.md` | uv, pnpm, Docker |
| Processes | `docs/invariants/processes.md` | Plans, ADRs, cadence, README |

## Domain Docs

| Doc | Scope |
|-----|-------|
| `docs/domain/genai-assignment/index.md` | Architecture, chosen libs, phase status, dataset config |

## Plan Files

| File | Scope |
|------|-------|
| `docs/plans/setup.md` | Shared infra: dataset, Optuna, MLflow/W&B, logging |
| `docs/plans/task1-universal-autoencoder.md` | Task 1: Universal denoising autoencoder |
| `docs/plans/task2-hard-routing.md` | Task 2: Classifier + specialist autoencoders |
| `docs/plans/task3-soft-moe.md` | Task 3: Soft mixture-of-experts |
| `docs/plans/task4-face-to-sketch.md` | Task 4: Conditional GAN face-to-sketch |
| `docs/plans/app.md` | Shared React/FastAPI/Docker application |
