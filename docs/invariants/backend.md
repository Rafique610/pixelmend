# Backend Invariants (FastAPI)

## Folder Structure

```
src/app/backend/
  main.py              # App creation, middleware, router includes — no logic
  config.py            # Pydantic Settings class, env-backed
  routers/
    health.py          # GET /health
    task1.py           # Universal restoration endpoints
    task2.py           # Hard-routed restoration endpoints
    task3.py           # Soft MoE restoration endpoints
    task4.py           # Face-to-sketch endpoints
  schemas/
    task1.py           # Request/response Pydantic models for task 1
    task2.py           # …
    task3.py
    task4.py
    common.py          # Shared schemas (image upload, error response)
  services/
    inference.py       # ONNX Runtime session management, model loading
    preprocessing.py   # Image validation, resize, normalization
```

## Separation of Concerns

- **Routers** are thin — wire endpoint to service call, return schema.
- **Schemas** define request/response shapes + validation. Every endpoint
  that accepts a body or meaningful params MUST have a schema.
- **Services** hold business logic (preprocessing, inference, timing).
- **No raw SQL** — this project has no DB for the backend (Optuna DB is
  training-side only).

## Endpoints (Minimum Required)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/api/v1/restore/universal` | Task 1 inference |
| POST | `/api/v1/restore/hard-routed` | Task 2 inference |
| POST | `/api/v1/restore/soft-moe` | Task 3 inference |
| POST | `/api/v1/sketch/generate` | Task 4 inference |

Each inference endpoint must return: result image (base64 or file), inference
time (ms), and task-specific metadata (corruption probabilities, routing
weights, etc.).

## ONNX Runtime Inference

- Load ONNX models at startup via lifespan event, not per-request.
- Use `onnxruntime.InferenceSession` with `CPUExecutionProvider` as default;
  add `CUDAExecutionProvider` when GPU is available.
- Models live in `models/onnx/` — paths configured via `Settings`.

## Configuration

- All config through `Settings(BaseSettings)` in `config.py`.
- Env prefix: `GENAI_`.
- Never read `os.environ` directly.

## CORS

- Allow `http://localhost:3000` (React dev) and `http://localhost:5173`
  (Vite dev) during development.
- In production (Docker), allow the frontend container's origin.
