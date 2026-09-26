# Application Backend (FastAPI)

## Overview
Thin, high-performance REST API serving all four generative AI tasks via ONNX Runtime sessions loaded at application lifespan startup.

## Structure
- `main.py`: Application factory, lifespan event handler (pre-loading ONNX models), CORS middleware, and router registration. No business logic.
- `config.py`: Pydantic `Settings` class with `GENAI_` prefix.
- `routers/`:
  - `health.py`: `GET /health` service and model readiness check.
  - `task1.py`: `POST /api/v1/restore/universal`
  - `task2.py`: `POST /api/v1/restore/hard-routed`
  - `task3.py`: `POST /api/v1/restore/soft-moe`
  - `task4.py`: `POST /api/v1/sketch/generate`
- `schemas/`: Pydantic `BaseModel` request and response contracts.
- `services/`:
  - `inference.py`: ONNX Runtime session manager and model caching.
  - `preprocessing.py`: Image validation, resizing ($128 \times 128$), normalization, and tensor conversions.
