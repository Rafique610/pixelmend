# Task 1: Universal Denoising Autoencoder — Implementation Plan

This plan documents the end-to-end design, implementation, tuning, evaluation, and export of a single universal denoising autoencoder for Task 1 of the Generative AI assignment.

## Context & Assignment Constraints

- **Objective**: Design and train a single convolutional autoencoder capable of blind image restoration across four conditions: clean images, salt-and-pepper noise, Gaussian blur, and rectangular occlusion. The model receives no metadata indicating which corruption is present.
- **Dataset**: Oxford-IIIT Pet dataset (37 cat and dog breeds), resized to 128×128 RGB, normalized to $[0, 1]$.
- **Data Split**: Official `trainval` split into 80% train / 20% validation (`random_seed=42`). Official test set remains untouched until final evaluation.
- **Architectural Rules**:
  - Convolutional encoder $\to$ compressed bottleneck $\to$ convolutional decoder.
  - Progressive spatial reduction with increasing channel depth in encoder; mirrored spatial expansion with decreasing channel depth in decoder.
  - Meaningful bottleneck compression: no unrestricted skip connections that allow the network to bypass bottleneck compression.
  - Any limited skip connections used must be investigated, evaluated, and justified against a plain bottleneck baseline.
- **Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_1(x, \hat{x}) + (1 - \alpha) \cdot (1 - \text{SSIM}(x, \hat{x}))$$
  Initial baseline weighting $\alpha = 0.8$; final value optimized via Optuna.
- **Hyperparameter Optimization**: SQLite-backed Optuna study exploring learning rate, batch size, bottleneck dimension, encoder channel capacity, dropout, and $\alpha$.
- **Evaluation Requirements**: Per-corruption and per-severity (low, medium, high) breakdown of PSNR, SSIM, and L1 metrics; 12+ representative qualitative examples with absolute error maps; 4+ analyzed failure cases.
- **Deployment Target**: ONNX graph ($\text{opset} \ge 17$) with dynamic batch axis, integrated into FastAPI endpoint `/api/v1/restore/universal` for the "Universal Restoration" workspace.

---

## Step 1: Architecture Research — Encoder-Decoder Design

### Decision & Why It Matters
The universal autoencoder must restore clean images and three distinct corruption modalities (high-frequency impulsive noise, low-pass Gaussian smoothing, and large spatial block occlusions) without corruption conditioning. The core architectural decision is choosing an encoder-bottleneck-decoder topology that enforces sufficient spatial compression to learn meaningful generative priors while retaining enough capacity to restore fine structural details across all corruption types.

The assignment mandates progressive spatial reduction with increasing channels and a meaningful bottleneck. Unrestricted skip connections (like standard U-Net) allow raw inputs to bypass the bottleneck, undermining compression. However, pure bottleneck models can struggle with high-frequency edge recovery. We must evaluate whether a plain convolutional stack, a residual-block encoder-decoder, or a U-Net-lite with strictly limited, bottlenecked skip connections best balances restoration quality with assignment compliance.

### Alternatives to Research

| Variant | Estimated Parameters | Skip Connections | Bottleneck Compression | Multi-Corruption Suitability | Implementation Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1: Plain Conv Stack** | ~1.2M – 2.0M | None (strict bottleneck) | High (e.g., $128\times 128 \to 8\times 8 \times 256$ or $4\times 4 \times 512$) | Baseline for blur/noise; struggles with sharp texture restoration under occlusion | Low (sequential Conv2d $\to$ BN $\to$ LeakyReLU $\to$ MaxPool; ConvTranspose2d upsampling) |
| **Option 2: ResBlock-based AE** | ~2.5M – 4.2M | Intra-block residual connections only; no encoder-to-decoder skips | High (same bottleneck spatial reduction, higher representation capacity) | Strong gradient propagation; proven in DnCNN and image restoration baselines | Moderate (residual convolution blocks with identity/projection shortcuts within stages) |
| **Option 3: U-Net-lite (Limited Skips)** | ~1.8M – 3.0M | 1–2 restricted skip connections with $1\times 1$ conv bottleneck / channel reduction | Moderate (compressed feature maps injected at intermediate decoder stage) | High detail preservation; risk of partially bypassing compression if unconstrained | Moderate-High (requires careful channel bottlenecking on skip paths to justify per assignment spec) |

### Open Question for User
- **Bottleneck Spatial Dimension & Latent Representation**: Whether to compress down to a spatial feature map (e.g., $8\times 8 \times 256$ or $4\times 4 \times 512$) or flatten into a 1D vector (e.g., 256-d or 512-d with a dense layer). Spatial bottlenecks preserve coarse topology for inpainting, whereas 1D vectors enforce maximum abstraction. This should remain tunable during implementation and Optuna search rather than locked early.

### Recommended Approach
*(To be filled during implementation after architecture prototyping and user review.)*

### Research Notes
*(Findings, parameter counts, and empirical validation observations to be recorded during implementation.)*

### Files Changed / Created
- `src/task1/encoder.py`
- `src/task1/decoder.py`
- `src/task1/autoencoder.py`

---

## Step 2: Loss Function Research — L1 + SSIM Weighting

### Decision & Why It Matters
The assignment specifies the composite reconstruction loss:
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_1(x, \hat{x}) + (1 - \alpha) \cdot (1 - \text{SSIM}(x, \hat{x}))$$
The weighting parameter $\alpha \in [0, 1]$ dictates the trade-off between pixel-level fidelity (L1 penalty) and structural coherence / luminance / contrast preservation (SSIM). Multi-corruption restoration creates competing objectives:
- Salt-and-pepper noise and occlusion create large localized pixel errors that L1 penalizes uniformly.
- Gaussian blur primarily attenuates high-frequency edges and gradients, which SSIM captures much more effectively than mean pixel error.
- Pure L1 loss often yields blurry reconstructions, while pure SSIM can introduce high-frequency checkerboard artifacts or slight chromatic shifts.
We must also evaluate whether adding an optional perceptual loss (e.g., VGG-16 feature reconstruction) offers demonstrable benefit without violating assignment simplicity constraints.

### Alternatives to Research

| Loss Formulation | Formula / Components | Advantages | Disadvantages / Risks |
| :--- | :--- | :--- | :--- |
| **Option 1: Pure L1 ($\alpha = 1.0$)** | $\mathcal{L}_1(x, \hat{x}) = \frac{1}{C\cdot H\cdot W}\sum \|x - \hat{x}\|$ | Numerically stable, monotonic gradient, penalizes outliers less than L2 | Prone to over-smoothing; fails to preserve sharp edge boundaries in deblurring |
| **Option 2: Pure SSIM ($\alpha = 0.0$)** | $1 - \text{SSIM}(x, \hat{x})$ | Strongly penalizes structural degradation and edge blurring | Can allow color shifts and localized intensity offsets; slower convergence |
| **Option 3: Combined L1 + SSIM ($\alpha = 0.8$ default)** | $0.8 \cdot \mathcal{L}_1 + 0.2 \cdot (1 - \text{SSIM})$ | Well-balanced default recommended by assignment; pairs pixel convergence with structural sharpness | Requires hyperparameter tuning across varied corruptions; fixed $\alpha$ may not be optimal for all severities |
| **Option 4: Combined + Perceptual Loss (Optional)** | $\alpha \mathcal{L}_1 + (1-\alpha)(1-\text{SSIM}) + \beta \mathcal{L}_{\text{VGG}}$ | Superior visual realism on semantic pet features (fur, eyes) | High computational overhead, requires pretrained VGG-16 backbone, adds extra hyperparameter $\beta$ |

### Side-by-Side Quick Experiment Protocol
Before committing to baseline training, execute a 5-epoch quick run on a 15% training subset (approx. 450 images) evaluated on a fixed validation subset for $\alpha \in \{0.0, 0.5, 0.8, 1.0\}$. Record validation PSNR, SSIM, and visible reconstruction artifacts.

| $\alpha$ Value | Description | Val Loss (5 ep) | Val PSNR (dB) | Val SSIM | Visual Artifacts & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\alpha = 0.0$ | Pure SSIM | *(TBD)* | *(TBD)* | *(TBD)* | *(To be recorded during experiment)* |
| $\alpha = 0.5$ | Equal weight L1 / SSIM | *(TBD)* | *(TBD)* | *(TBD)* | *(To be recorded during experiment)* |
| $\alpha = 0.8$ | Assignment default | *(TBD)* | *(TBD)* | *(TBD)* | *(To be recorded during experiment)* |
| $\alpha = 1.0$ | Pure L1 | *(TBD)* | *(TBD)* | *(TBD)* | *(To be recorded during experiment)* |

### Recommended Approach
*(To be filled based on quick experiment results.)*

### Research Notes
*(Findings and loss stability observations to be recorded during implementation.)*

### Files Changed / Created
- `src/shared/losses.py`
- `src/task1/train.py`

---

## Step 3: Implement Universal Autoencoder

### Scope
Translate the selected Step 1 architecture and Step 2 loss formulation into modular PyTorch components. Construct the end-to-end model, dynamic data augmentation corruption pipeline, deterministic validation loop, and checkpointing infrastructure.

### What to Build
1. **Encoder Module (`src/task1/encoder.py`)**:
   - Input: $(B, 3, 128, 128)$ RGB image tensor normalized to $[0, 1]$.
   - Progressive spatial downsampling stages: $128\times 128 \to 64\times 64 \to 32\times 32 \to 16\times 16 \to 8\times 8$ (or $4\times 4$).
   - Channel progression: e.g., $3 \to 32 \to 64 \to 128 \to 256$ (configurable via constructor arguments).
   - Downsampling via strided convolutions or MaxPool2d, paired with Batch Normalization (or LayerNorm) and LeakyReLU activations.
   - Bottleneck projection layer with configurable latent capacity and optional dropout.
2. **Decoder Module (`src/task1/decoder.py`)**:
   - Progressive spatial upsampling stages: mirrors encoder geometry back to $(B, 3, 128, 128)$.
   - Upsampling via `ConvTranspose2d` (or `Upsample(scale_factor=2)` followed by `Conv2d` to prevent checkerboard artifacts).
   - Final projection layer: $1\times 1$ convolution mapping to 3 channels with `Sigmoid` activation to constrain outputs strictly to $[0, 1]$.
3. **Universal Autoencoder Wrapper (`src/task1/autoencoder.py`)**:
   - Encapsulates encoder, bottleneck, and decoder into a single `nn.Module`.
   - Forward pass accepts corrupted input $x_{\text{corrupt}}$ and returns restored output $\hat{x}$.
   - Provides property helpers for parameter counting and bottleneck compression ratio calculation.
4. **Training Script (`src/task1/train.py`)**:
   - Integrated training loop: draws clean samples from the 80% train split, applies on-the-fly random corruptions (clean, salt-and-pepper, Gaussian blur, rectangular occlusion) with equal probability.
   - Computes composite loss $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_1 + (1-\alpha)(1-\text{SSIM})$.
   - Deterministic validation loop: iterates over pre-generated validation manifest (`manifests/val_manifest.json`), computing per-corruption and aggregate PSNR, SSIM, and L1 metrics.
   - Logs epoch metrics to experiment tracker (MLflow or W&B).
   - Checkpointing: saves best checkpoint (`checkpoints/task1/best_model.pth`) and latest checkpoint containing `model.state_dict()`, `optimizer.state_dict()`, epoch index, and validation metrics.

### Key Architectural & Pipeline Details
- Device management routed through `get_device()` in `src/shared/config.py`.
- Shared losses imported from `src/shared/losses.py`; metrics imported from `src/shared/metrics.py`.
- Mixed precision support via `torch.amp.autocast('cuda')` and `GradScaler`.
- DataLoader configuration: `num_workers >= 2`, `pin_memory=True`, `persistent_workers=True`.

### Verification
- **Shape Verification**: Run dummy tensor $(2, 3, 128, 128)$ through model forward pass; verify output shape is exactly $(2, 3, 128, 128)$ with values in $[0, 1]$.
- **Gradient Flow Check**: Run a single forward/backward step; assert all module parameters have non-zero, non-NaN gradients.
- **Overfit Smoke Test**: Train on a tiny batch of 4 images for 20 iterations; verify loss monotonically decreases toward 0.
- **Validation Determinism Check**: Run validation loop twice on `manifests/val_manifest.json`; assert identical metrics across runs.

### Files Changed / Created
- `src/task1/encoder.py`
- `src/task1/decoder.py`
- `src/task1/autoencoder.py`
- `src/task1/train.py`

---

## Step 4: Training & Validation Run (Baseline)

### Scope
Conduct full baseline training of the Universal Autoencoder prior to hyperparameter optimization. Establish reference benchmark metrics and convergence dynamics on the complete Oxford-IIIT Pet training split.

### What to Build
- Execute full training run using chosen baseline configuration (initial $\alpha = 0.8$, standard channel progression $[32, 64, 128, 256]$, AdamW optimizer with initial learning rate $1\times 10^{-3}$, cosine decay scheduler, batch size 32).
- Train for 50–100 epochs with early stopping patience of 10 epochs monitoring validation loss.
- Track metrics every epoch in experiment tracker (`genai-task1-universal-ae`):
  - Training loss ($\mathcal{L}_{\text{total}}$, $\mathcal{L}_1$, $\mathcal{L}_{\text{SSIM}}$)
  - Validation loss (aggregate and per corruption type)
  - Validation PSNR and SSIM
- Image Reconstruction Sampling:
  - Every 5 epochs, generate a fixed visual sample grid of 8 images from the validation set (2 clean, 2 salt-and-pepper, 2 blur, 2 occlusion).
  - Display side-by-side: Ground Truth Clean $\mid$ Corrupted Input $\mid$ Model Reconstruction $\mid$ Absolute Error Map.
  - Log sample grid artifact to experiment tracker.
- Persist best baseline weights to `checkpoints/task1/baseline_best.pth`.
- Export training curves (loss curves, PSNR/SSIM trajectories) to disk and tracking dashboard.

### Verification
- Loss curves demonstrate stable convergence without divergence or oscillations.
- Validation PSNR and SSIM plateau at competitive baseline values.
- Visual reconstructions at epoch 30+ show noticeable noise suppression, deblurring, and occlusion inpainting compared to corrupted inputs.
- Checkpoint file `checkpoints/task1/baseline_best.pth` exists, is non-empty, and successfully reloads into the model class.

### Files Changed / Created
- `checkpoints/task1/baseline_best.pth`
- Experiment tracking run logs and training curve artifacts.

---

## Step 5: Optuna Hyperparameter Search

### Scope
Implement automated Bayesian hyperparameter optimization using Optuna to maximize restoration quality across all four image conditions simultaneously.

### Optuna Study Specifications

| Configuration Field | Specification | Rationale |
| :--- | :--- | :--- |
| **Study Name** | `task1-universal-ae` | Aligns with project ML invariants naming convention |
| **Storage Backend** | SQLite database at `optuna/optuna_studies.db` | Persists across process restarts and system reboots |
| **Direction** | `minimize` (validation loss) | Minimizing $\mathcal{L}_{\text{val}} = \alpha \mathcal{L}_1 + (1-\alpha)(1-\text{SSIM})$ directly optimizes the joint objective. (Alternatively `maximize` on validation SSIM; minimizing validation loss with normalized metric scaling prevents metric scale imbalance). |
| **Pruner** | `MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)` | Prunes unpromising trials early once initial warm-up epochs indicate subpar convergence |
| **Number of Trials** | 30–50 trials | Sufficient coverage of 6-dimensional search space within compute budget |

### Hyperparameter Search Space

| Hyperparameter | Distribution Type | Search Range / Choices | Description |
| :--- | :--- | :--- | :--- |
| `learning_rate` | Log-uniform | $[1\times 10^{-4},\, 1\times 10^{-2}]$ | Base learning rate for AdamW optimizer |
| `batch_size` | Categorical | $\{16, 32, 64\}$ | Mini-batch size affecting gradient noise and throughput |
| `bottleneck_dim` | Categorical | $\{64, 128, 256, 512\}$ | Latent channel capacity / feature compression depth |
| `encoder_channels` | Categorical | `[32, 64, 128]`, `[64, 128, 256]`, `[32, 64, 128, 256]` | Width progression controlling model capacity |
| `dropout` | Uniform | $[0.0,\, 0.5]$ | Dropout rate applied at bottleneck layer to prevent co-adaptation |
| `alpha` | Uniform | $[0.5,\, 1.0]$ | Weight of L1 loss relative to $(1 - \text{SSIM})$ in objective function |

### Implementation Protocol
1. **Optuna Script (`src/task1/optuna_search.py`)**:
   - Instantiates or loads existing SQLite study from `optuna/optuna_studies.db`.
   - Defines `objective(trial)` function:
     - Samples hyperparameters from specified search space.
     - Instantiates model, optimizer, scheduler, and loss function with trial configurations.
     - Runs training loop for up to 30 epochs per trial.
     - At the end of each epoch, evaluates on validation manifest, calls `trial.report(val_loss, step=epoch)`.
     - Checks `if trial.should_prune(): raise optuna.TrialPruned()`.
     - Logs trial parameters and intermediate metrics to experiment tracker under run name `trial-{trial.number}`.
     - Returns final validation loss as optimization objective.
2. **Post-Search Analysis**:
   - Prints best trial number, optimal objective score, and parameter configuration dictionary.
   - Generates hyperparameter importances plot (`optuna.visualization.plot_param_importances`) and optimization history plot (`optuna.visualization.plot_optimization_history`).

### Verification
- Run 2-trial test run with 2 epochs each to verify pruning hooks and SQLite writes operate cleanly.
- Verify trials are visible in SQLite database via sqlite3 query and experiment tracking UI.
- Ensure best parameter set is printed and serialized to `results/task1/best_hyperparams.json`.

### Files Changed / Created
- `src/task1/optuna_search.py`
- `optuna/optuna_studies.db`
- `results/task1/best_hyperparams.json`

---

## Step 6: Final Retrain with Best Config

### Scope
Train the definitive Universal Autoencoder model from scratch using the winning hyperparameter configuration discovered in Step 5 for a full training schedule.

### What to Build
- Script execution mode in `src/task1/train.py` taking configuration from `results/task1/best_hyperparams.json`.
- Train model from scratch on full 80% training split for the complete schedule (e.g., 80–100 epochs, cosine annealing learning rate scheduler with 5-epoch warm-up).
- Log run in experiment tracker with designated run name `final`.
- Save best model weights to canonical location `checkpoints/task1/best_model.pth`.
- Checkpoint payload must contain:
  - `model_state_dict`: Model weights
  - `optimizer_state_dict`: Optimizer state
  - `epoch`: Best validation epoch index
  - `best_val_loss`: Best aggregate validation loss
  - `config`: Complete hyperparameter dictionary
- Full validation assessment: evaluate best model on deterministic validation manifest, populating a preliminary per-corruption, per-severity metric summary table.

### Verification
- Final retrained model validation loss and SSIM match or improve upon the best metric reported in the corresponding Optuna trial.
- Weight file `checkpoints/task1/best_model.pth` exists, is valid, and loads into `UniversalAutoencoder` with `strict=True`.
- Tracking platform shows completed `final` run with all associated metrics and training curves.

### Files Changed / Created
- `checkpoints/task1/best_model.pth`
- `src/task1/train.py`

---

## Step 7: Evaluation & Visual Results

### Scope
Perform exhaustive quantitative and qualitative evaluation of the final universal autoencoder on the official test set using the deterministic test manifest (`manifests/test_manifest.json`). Produce all tables, comparative figures, and failure case analyses required by the assignment specification.

### What to Build
1. **Evaluation Script (`src/task1/evaluate.py`)**:
   - Loads `checkpoints/task1/best_model.pth` and sets model to evaluation mode (`eval()`, `torch.no_grad()`).
   - Loads deterministic test manifest specifying image paths, corruption types, and exact severity parameters.
   - Computes PSNR, SSIM, and L1 reconstruction error across all test samples.
2. **Quantitative Breakdown Tables**:
   - **Per-Corruption Summary Table**:
     - Columns: Corruption Type $\mid$ Mean PSNR (dB) $\mid$ Mean SSIM $\mid$ Mean L1 Error.
     - Rows: Clean, Salt-and-Pepper, Gaussian Blur, Rectangular Occlusion, Overall Mean.
   - **Per-Severity Breakdown Table**:
     - Columns: Corruption $\mid$ Severity Level (Low / Medium / High) $\mid$ Severity Parameters $\mid$ PSNR (dB) $\mid$ SSIM $\mid$ L1 Error.
     - Covers all 9 corruption $\times$ severity combinations defined in domain index:
       - Salt-and-pepper: Low ($p=0.03$), Medium ($p=0.08$), High ($p=0.15$)
       - Gaussian blur: Low ($k=3, \sigma=0.7$), Medium ($k=5, \sigma=1.5$), High ($k=7, \sigma=2.5$)
       - Occlusion: Low ($\sim 10\%$ area, 1 rect), Medium ($\sim 20\%$ area, 2 rects), High ($\sim 35\%$ area, 3 rects)
3. **Qualitative Visual Figures (12+ Representative Samples)**:
   - Select 12+ diverse pet test images spanning cat and dog breeds, corruption types, and severity levels (at least 3 samples per corruption condition).
   - Render 4-panel figures for each example:
     $$\text{Ground Truth Clean} \quad\mid\quad \text{Corrupted Input} \quad\mid\quad \text{Reconstructed Output} \quad\mid\quad \text{Absolute Error Map } (|x - \hat{x}|)$$
   - Colorize error maps using a consistent colormap (e.g., `inferno` or `jet`) with fixed scale $[0, 0.5]$ to enable cross-comparison.
4. **Failure Case Analysis (4+ Distinct Cases)**:
   - Identify at least 4 test instances where reconstruction quality degrades severely (e.g., lowest PSNR/SSIM).
   - Detail specific failure modes:
     - Severe rectangular occlusion across discriminative facial landmarks (eyes, snout) resulting in hallucinated or asymmetric features.
     - High-density salt-and-pepper noise ($p=0.15$) leading to chromatic desaturation or mottled textures.
     - Large-kernel Gaussian blur ($k=7, \sigma=2.5$) causing loss of fine fur texture and soft contour edges.
     - Complex background occlusion leading to boundary bleeding between pet silhouette and background.
   - Provide visual figure and analytical explanation of architectural/capacity limitations contributing to each failure.
5. **Output Serialization**:
   - Save structured numerical results to `results/task1/metrics_summary.json`.
   - Save figures to `results/task1/visualizations/` and log all artifacts to the experiment tracker.

### Verification
- Metrics are calculated from real test set evaluations; no mock or placeholder values.
- 12+ representative qualitative figures and 4+ failure case figures generated and saved as high-resolution PNGs.
- Error maps accurately reflect the pixel difference magnitude between clean targets and reconstructions.

### Files Changed / Created
- `src/task1/evaluate.py`
- `results/task1/metrics_summary.json`
- `results/task1/visualizations/representative_examples/`
- `results/task1/visualizations/failure_cases/`

---

## Step 8: ONNX Export & Verification

### Scope
Export the final trained PyTorch universal autoencoder to an optimized ONNX computational graph, verify numerical parity against PyTorch, and prepare the artifact for integration into the FastAPI "Universal Restoration" workspace.

### Export Specifications
- **Source Model**: `checkpoints/task1/best_model.pth` loaded into `UniversalAutoencoder`.
- **Target Export Path**: `models/onnx/task1_universal_ae.onnx`.
- **Target Application Workspace**: Workspace 1 ("Universal Restoration"), backend endpoint `/api/v1/restore/universal`.
- **ONNX Opset Version**: $\text{opset\_version} \ge 17$.
- **Tensor Specifications**:
  - Input Tensor: Name `'input'`, shape `(batch_size, 3, 128, 128)`, dtype `float32`, dynamic batch axis.
  - Output Tensor: Name `'output'`, shape `(batch_size, 3, 128, 128)`, dtype `float32`, dynamic batch axis.
- **Dynamic Axes Map**:
  ```python
  dynamic_axes = {
      'input': {0: 'batch_size'},
      'output': {0: 'batch_size'}
  }
  ```

### Verification Procedure
1. **Export Script (`src/task1/export_onnx.py`)**:
   - Sets model to `eval()` mode.
   - Exports graph using `torch.onnx.export()` with `do_constant_folding=True`.
   - Validates ONNX graph structure using `onnx.checker.check_model()`.
2. **Numerical Parity Assertion**:
   - Generates a fixed validation batch $x_{\text{val}} \in \mathbb{R}^{4\times 3\times 128\times 128}$.
   - Runs forward pass through PyTorch model: $y_{\text{pytorch}} = \text{model}(x_{\text{val}})$.
   - Initializes ONNX Runtime session: `ort_session = ort.InferenceSession('models/onnx/task1_universal_ae.onnx')`.
   - Runs inference through ONNX Runtime: $y_{\text{ort}} = \text{ort\_session.run(None, {'input': x_{\text{val}}.numpy()})}[0]$.
   - Asserts numerical equivalence:
     $$\text{np.allclose}(y_{\text{pytorch}}\text{.cpu().numpy()}, y_{\text{ort}}, \text{atol}=1\times 10^{-5})$$
   - Computes and logs the maximum absolute error:
     $$\Delta_{\max} = \max |y_{\text{pytorch}} - y_{\text{ort}}|$$
3. **Inference Latency Benchmark**:
   - Runs 100 warm-up iterations followed by 200 timed inference passes on single-image input $(1, 3, 128, 128)$ for both PyTorch and ONNX Runtime.
   - Logs mean latency (ms) and throughput (images/sec) to verify inference efficiency for the FastAPI backend.
4. **Tracker Logging**:
   - Log export run as `onnx-verify` in experiment tracker with parity metrics and latency numbers.

### Verification
- `models/onnx/task1_universal_ae.onnx` successfully created and passes `onnx.checker.check_model()`.
- Numerical parity assertion passes with $\Delta_{\max} < 1\times 10^{-5}$.
- ONNX Runtime executes inference with dynamic batch sizes (testing $B=1$, $B=4$, and $B=8$).

### Files Changed / Created
- `src/task1/export_onnx.py`
- `models/onnx/task1_universal_ae.onnx`
