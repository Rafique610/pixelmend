# ML / Training Invariants (PyTorch, Optuna, MLflow/W&B, ONNX)

Rules for all model training, hyperparameter optimization, experiment
tracking, and export in this project.

## PyTorch

- **Version**: PyTorch ≥ 2.1 with CUDA support where available.
- **Device handling**: Use a single `get_device()` helper in `src/shared/config.py`.
  Never scatter `torch.device(...)` calls across files.
- **Reproducibility**: Set `torch.manual_seed`, `torch.cuda.manual_seed_all`,
  `torch.backends.cudnn.deterministic = True`, and
  `torch.backends.cudnn.benchmark = False` for reproducible runs. Use
  `benchmark = True` only for speed-focused final training (document it).
- **Data loading**: Always use `num_workers ≥ 2` and `pin_memory=True` when
  CUDA is available. Use `persistent_workers=True` for multi-epoch runs.
- **Mixed precision**: Use `torch.amp.autocast` and `GradScaler` for
  training runs > 20 epochs. Document the precision choice.
- **Checkpointing**: Save `model.state_dict()`, `optimizer.state_dict()`,
  epoch, and best metric. Use `torch.save()`; one checkpoint per best-so-far
  plus a latest checkpoint.
- **Model definitions**: One file per logical unit (encoder, decoder, gate,
  discriminator). Max ~200 lines per model file.

## Optuna

- **Storage backend**: SQLite database at `optuna/optuna_studies.db` (shared
  across all tasks). Never use in-memory storage — studies must survive
  restarts.
- **Study naming convention**: `task{N}-{description}` — e.g.
  `task1-universal-ae`, `task2-classifier`, `task2-specialists`,
  `task3-moe-joint`, `task4-cgan`.
- **Each task gets its own Optuna step** — not a bullet inside a bigger step.
- **Direction**: `minimize` for loss-based objectives, `maximize` for
  metric-based. Document which.
- **Pruning**: Use `MedianPruner` or `HyperbandPruner`. Report pruner
  choice and why.
- **Trial reporting**: Log intermediate values every epoch via
  `trial.report()` + `trial.should_prune()`.
- **Logging**: Log every trial's params and final value to MLflow/W&B as
  well, so the tracking UI shows the full search.

## MLflow / Weights & Biases

- **MLflow** (local, self-hosted) is the chosen tracker for this project.
- Local tracking server with `mlruns/` directory (gitignored).
- **Experiment naming**: `genai-task{N}-{description}` — e.g.
  `genai-task1-universal-ae`.
- **What to log per run**:
  - All hyperparameters (as params).
  - Training loss per epoch (as metrics).
  - Validation loss + PSNR + SSIM per epoch (as metrics).
  - Best checkpoint path (as artifact or param).
  - Sample images at fixed intervals (as artifacts/images).
  - Optuna trial ID when the run is part of a search.
  - Final test metrics (as metrics, logged once).
- **Run naming**: `trial-{N}` for Optuna trials, `final` for the selected
  best config retrain, `onnx-verify` for the ONNX comparison run.
- **Never log model weights as artifacts** unless explicitly needed —
  checkpoints go to `checkpoints/` on disk.

## ONNX Export

- **Each task gets its own ONNX export step** — not a bullet inside a bigger
  step.
- **Export function**: Use `torch.onnx.export()` with `opset_version ≥ 17`.
- **Dynamic axes**: Always specify dynamic batch dimension.
- **Verification**: After export, run the same validation batch through both
  PyTorch and ONNX Runtime. Assert `np.allclose(pytorch_out, ort_out,
  atol=1e-5)`. Log the max absolute difference.
- **Output path**: `models/onnx/task{N}_{model_name}.onnx`.
- **Models to export**:
  - Task 1: universal autoencoder.
  - Task 2: classifier + 3 specialist autoencoders (4 files).
  - Task 3: full soft MoE pipeline (gate + experts as one graph, or
    separate — research which is better for inference).
  - Task 4: generator only (discriminator is training-only).

## Shared Training Conventions

- **Loss functions**: Always implement losses in `src/shared/losses.py` when
  reused across tasks (L1, SSIM, combined reconstruction loss). Task-specific
  losses live in the task folder.
- **Evaluation metrics**: PSNR, SSIM, L1 error. Implemented once in
  `src/shared/metrics.py`.
- **Image format**: All images RGB, 128×128, normalized to [0, 1] float32.
- **Batch logging**: Log sample reconstructions every N epochs (N configurable,
  default 5) using the same fixed validation images so visual progress is
  comparable.
- **Early stopping**: Optional but document the patience and delta if used.

## Directory Layout for ML Artifacts

```
data/                  # downloaded datasets (gitignored)
optuna/                # Optuna SQLite DB
checkpoints/           # model checkpoints (gitignored, provide download link)
  task1/
  task2/
  task3/
  task4/
models/
  onnx/                # exported ONNX models
manifests/             # deterministic corruption manifests (JSON)
```
