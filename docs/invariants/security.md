# Security Invariants

## Credentials & Env Vars

- **Never hardcode credentials** — not in code, not in Docker Compose, not
  in config files. Source from `.env` / `.env.local`.
- **Never read `os.environ` directly** — always through Pydantic `Settings`.
- **Env prefix**: `GENAI_` for all project-specific vars.
- Docker Compose references `${VAR}` for build-time, `$${VAR}` for
  container-runtime env in healthchecks.

## API Keys

- If using W&B: `WANDB_API_KEY` in `.env`, never in code.
- If using MLflow remote: tracking URI in `.env`.
- No API keys in git history. `.env` is gitignored.

## File Upload Validation

- Backend must validate uploaded file type (JPEG/PNG only), file size
  (max 10 MB), and image dimensions before processing.
- Never pass user-uploaded filenames to the filesystem — generate UUIDs.

## Model Files

- ONNX models do not contain credentials, but they should not be committed
  to git if > 100 MB. Use Git LFS or provide download links.
