# Tooling Invariants

## Package Managers

| Stack | Manager | Lock file |
|-------|---------|-----------|
| Python (training + backend) | `uv` | `uv.lock` |
| Node (frontend) | `pnpm` | `pnpm-lock.yaml` |

- Never call `python` directly for project commands — use `uv run`.
- Never call `npm` or `yarn` — use `pnpm`.
- Always have `pyproject.toml` + `uv.lock` for Python.
- Always have `package.json` + `pnpm-lock.yaml` for Node.

## Docker

- **Two-stage Dockerfiles** — builder stage + slim runtime stage.
- **Docker Compose** starts the full app with one command:
  `docker compose up --build`.
- Frontend and backend are separate services in Compose.
- DNS failures: retry at least 3 times before giving up.

## Taskfile

- Provide a `Taskfile.yml` with common commands:
  - `task setup` — install all dependencies
  - `task download-data` — download datasets
  - `task train-task1` / `task train-task2` / etc.
  - `task optuna-task1` / etc.
  - `task export-onnx`
  - `task dev-backend` / `task dev-frontend`
  - `task docker-up` / `task docker-down`

## Paths

- All paths relative to project root.
- Use `pathlib.Path` in Python, never string concatenation for paths.
- Data directory: `data/` (gitignored).
- Checkpoints: `checkpoints/` (gitignored, provide download link).
- ONNX models: `models/onnx/`.

## Gitignore Maintenance (Mandatory Side-by-Side)

- Maintain `.gitignore` continuously alongside every step, feature, or new tool.
- Never allow large dataset archives, binary weights, MLflow logs, or local `.env` files to be tracked.
- Keep deterministic manifests (`manifests/*.json`) and JSON quantitative summaries tracked while ignoring large raw image outputs (`results/**/*.png`).
- `.memory/` is private and strictly gitignored.
