# Core Invariants

Rules that hold across every task in this project, regardless of stack.

## Folder Structure (Feature-Based)

```
src/
  shared/          # dataset, corruption, logging, config — used by all tasks
  task1/           # universal autoencoder
  task2/           # classifier + specialist autoencoders
  task3/           # soft mixture-of-experts
  task4/           # conditional GAN face-to-sketch
  app/
    backend/       # FastAPI
    frontend/      # React + Tailwind
```

- Every task gets its own top-level folder under `src/`.
- Shared utilities (dataset loading, corruption pipeline, logging helpers,
  Optuna/tracking wrappers) live in `src/shared/`.
- Never use horizontal technical layering (`src/models/`, `src/losses/`).
  Group all components of a task inside that task's directory.

## File-Size Limits

- No Python file over ~300 lines. Split or use a design pattern.
- No model definition over ~200 lines. Split encoder/decoder/gate into
  separate files within the task folder.

## Real Data Only & Empirical Integrity

Never show fabricated metrics, mock losses, invented accuracy numbers, or
placeholder images. Every number shown to the user must come from a real
computation — a real training run, a real evaluation, a real Optuna trial.

- **Explicit execution**: Whenever executing benchmarks, parameter sweeps,
  or training runs, execute the exact number of epochs and iterations claimed.
  Never assume, extrapolate, or misreport run depth.
- **Explicit storage**: Explicitly persist all experiment results, loss logs,
  and comparative visualizations to disk (JSON/figures) and tracking stores
  (MLflow) during or immediately following the run so all data is tangible and
  reproducible.

## Ponytail Ladder (Active for All Code)

1. Does this need to exist at all? (YAGNI)
2. Already in this codebase? Reuse.
3. Standard library does it? Use it.
4. Native platform feature? Use it.
5. Already-installed dependency? Use it.
6. Can it be one line? Make it one line.
7. Only then: write the minimum code that works.

No factories for one product. No speculative interfaces. Shortest working
diff wins.

## Research-First Design Constraint

This assignment is graded on research depth. Every important technical
decision (architecture variant, loss function, corruption strategy,
embedding approach, evaluation metric) must:

1. State the decision and why it matters.
2. Research 2–3 real alternatives with citations.
3. Summarize tradeoffs in a comparison table.
4. Where cheap, run a side-by-side and log real numbers.
5. Give a recommendation tied to evidence.
6. Write findings into the step's Research Notes for the IEEE report.

Never just pick an approach and build it. The investigation _is_ the
deliverable.

## Reproducibility

- All random seeds must be explicit and documented (dataset split: 42).
- Every training run must log its full config to MLflow/W&B.
- Deterministic validation/test corruption manifests — generated once,
  reused everywhere.
