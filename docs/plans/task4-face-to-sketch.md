# Task 4: Face-to-Sketch Generation System (Conditional GAN)

## Overview & Context

Task 4 builds an end-to-end conditional Generative Adversarial Network (cGAN) for paired facial sketch synthesis using the **FS2K (Facial Sketch Synthesis 2K)** dataset. The system receives a 128×128 RGB facial photograph $x$ and a target style category $s \in \{0, 1, 2\}$, generating a synthesized facial sketch $\hat{y} = G(x, s)$ in the requested artistic style.

### Dataset & Splits
- **Dataset**: FS2K dataset containing 2,104 paired high-resolution face photos and corresponding sketches across 3 distinct sketch styles.
- **Image Specifications**: Resized to 128×128 pixels, RGB photo input, paired sketches normalized to $[-1, 1]$ (or $[0, 1]$).
- **Split Strategy**: Official FS2K train/test partition. From the training partition, a deterministic 15% validation split is reserved (`random_seed=42`), stratified by sketch style to preserve identical style distributions across splits.
- **Pair Consistency**: Photo and sketch pairs are strictly coupled across all data loading and augmentation steps.

### Formulation & Losses
- **Generator**: U-Net encoder-decoder network conditioned on learned style embedding $e_s$: $\hat{y} = G(x, s)$.
- **Discriminator**: PatchGAN $D(x, y, s)$ conditioning on photo $x$, sketch candidate $y$ (real $y$ or fake $\hat{y}$), and style $s$.
- **Adversarial Objective**:
  $$\mathcal{L}_{\text{adv}}(G, D) = \mathbb{E}_{x, y, s} \left[\log D(x, y, s)\right] + \mathbb{E}_{x, s} \left[\log (1 - D(x, G(x, s), s))\right]$$
  Implemented using Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **Reconstruction Objective**:
  $$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, s} \left[ \| y - G(x, s) \|_1 \right]$$
- **Total Generator Loss**:
  $$\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \lambda_{L1} \cdot \mathcal{L}_{L1}$$
  with initial baseline weighting $\lambda_{L1} = 100$.
- **Discriminator Objective**:
  $$\mathcal{L}_D = \frac{1}{2} \left( \mathbb{E}_{x, y, s} \left[\text{BCE}(D(x, y, s), 1)\right] + \mathbb{E}_{x, s} \left[\text{BCE}(D(x, G(x, s), s), 0)\right] \right)$$

### Application & Deployment Context
- **Deployment**: Generator-only export to ONNX (`models/onnx/task4_generator.onnx`, opset $\ge 17$) with dynamic batch inference via ONNX Runtime.
- **App Workspace**: 'Face-to-Sketch Generator' (`/api/v1/sketch/generate`), supporting image upload / webcam input, interactive style selection (Style 1, 2, or 3), side-by-side comparative visualization, and sketch download.

---

## Step 1: Generator Architecture Research

### Decision & Why It Matters
Which U-Net variant to use for the generator $G(x, s)$. The generator must translate photographic color textures into crisp, stylized pencil strokes while preserving facial geometry (eyes, nose, mouth alignment, face contours). The architecture determines parameter footprint, gradient flow stability during adversarial training, and detail preservation across skip connections.

### Alternatives to Research

| Dimension | Vanilla U-Net (pix2pix) | ResNet-based U-Net | Attention U-Net |
| :--- | :--- | :--- | :--- |
| **Description** | Standard encoder-decoder with downsampling conv blocks, upsampling conv blocks (transposed conv / bilinear upsampling), and direct skip connections from encoder. | Replaces plain conv blocks with residual blocks (conv-norm-relu-conv + skip addition) in encoder, decoder, or bottleneck. | Adds attention gates at skip connections to dynamically weight spatial feature transfer from encoder to decoder. |
| **Params** | Low (~10M–15M params at base channels 64). | Moderate (~16M–22M params). | Moderate-High (~18M–25M params). |
| **Skip Connections** | Direct channel-wise concatenation across symmetric stages. | Direct concatenation of residual feature maps. | Attention-gated concatenation (suppresses irrelevant background regions). |
| **Gradient Flow** | Good through long skip connections; can degrade in deeper plain conv stacks. | Superior due to identity shortcut paths within residual blocks. | Good; attention coefficients introduce non-linear gating dynamics. |
| **Detail Preservation** | Strong baseline; retains low-level edges directly from encoder. | High; residual blocks retain fine stroke textures without degradation. | Highest; dynamically focuses on salient facial features (eyes, lips, contours). |
| **Complexity** | Low (standard reference pix2pix architecture). | Moderate (residual block wiring and channel matching). | High (attention gate modules, compatibility considerations for ONNX). |
| **Papers** | Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (CVPR 2017). | Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (ICCV 2017). | Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas* (MIDL 2018). |

### Recommended Approach
*(To be filled during implementation)*

### Research Notes
*(Empty — to be populated during implementation)*

---

## Step 2: Style Conditioning Research

### Decision & Why It Matters
How to inject the learned categorical style embedding into both the Generator $G$ and Discriminator $D$. In the FS2K dataset, the 3 sketch styles possess distinct artistic traits (varying stroke weights, shading densities, cross-hatching styles, and contour emphasis). Conditioning must exert strong, distinct stylistic control without destabilizing adversarial dynamics.

### Alternatives to Research

| Conditioning Method | Style Control Strength | Implementation Effort | Where to Inject | Papers |
| :--- | :--- | :--- | :--- | :--- |
| **1. Spatial Concatenation** | Moderate; can get diluted in deep intermediate layers. | Very low (lookup vector, spatially replicate to $H \times W$, concatenate). | Concatenated to input RGB image and/or bottleneck feature maps in G and D. | Isola et al. (2017); Mirza & Osindero, *Conditional Generative Adversarial Nets* (2014). |
| **2. FiLM (Feature-wise Linear Modulation)** | High; affine scale $\gamma(s)$ and shift $\beta(s)$ dynamically modulate feature maps. | Moderate (small MLP maps style embedding to per-channel $[\gamma, \beta]$). | Applied after normalization layers in generator decoder blocks and discriminator layers. | Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (AAAI 2018). |
| **3. AdaIN (Adaptive Instance Normalization)** | Very High; standardizes feature maps to style-derived mean and variance. | Moderate-High (custom normalization layer, requires careful numerical bounds). | Replaces standard normalization layers in bottleneck and decoder upsampling stages. | Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017). |
| **4. Class-Conditional Batch Normalization (CCBN) / Cond-IN** | High; discrete categorical embedding directly selects class-specific affine parameters. | Low-Moderate (embedding table replaces scalar affine weights and biases). | Normalization layers throughout all generator down/up blocks and D conv layers. | Dumoulin et al. (2017); Brock et al., *Large Scale GAN Training for High Fidelity Natural Image Synthesis* (BigGAN, ICLR 2019). |

### Open Question for User
- **Style Embedding Dimension ($d_s$)**: What range of embedding dimensions should be explored? With 3 discrete styles, typical choices are 8, 16, or 32 dimensions. Optuna will search within this specified range in Step 7.

### Recommended Approach
*(To be filled during implementation)*

### Research Notes
*(Empty — to be populated during implementation)*

---

## Step 3: Discriminator Architecture Research

### Decision & Why It Matters
Which PatchGAN variant to deploy as the discriminator $D(x, y, s)$. Unlike standard GAN discriminators that produce a single scalar score for the entire image, PatchGAN classifies whether local $N \times N$ overlapping patches are real or synthetic. On 128×128 images, the effective receptive field (RF) size dictates whether the discriminator forces fine stroke fidelity (local texture) or enforces overall facial geometry (global coherence).

### Alternatives to Research

| Variant | Receptive Field (RF) Size | Conv Layers | Global vs. Local Focus | Compute / Memory | Papers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 70×70 PatchGAN (pix2pix default)** | $\sim 70 \times 70$ pixels | 5-layer conv stack (stride-2 downsampling) | Balances facial geometry and local stroke rendering; covers $>50\%$ of 128×128 image. | Baseline standard; modest VRAM footprint. | Isola et al. (2017). |
| **2. 16×16 PatchGAN (smaller RF)** | $\sim 16 \times 16$ pixels | 3-layer conv stack (shallower downsampling) | Focuses strictly on fine stroke sharpness and local texture; relies on L1 loss for global structure. | Extremely lightweight; fast training iteration. | Li & Wand, *Precomputed Real-Time Synthesizing of Style* (ECCV 2016). |
| **3. Multi-Scale Discriminator** | Dual RF: $70 \times 70$ (full scale) + coarse RF (half scale) | Two parallel PatchGANs ($D_1$ on $128 \times 128$, $D_2$ on downsampled $64 \times 64$) | Simultaneous enforcement of fine sketch stroke fidelity and global facial symmetry. | $\sim 2\times$ compute and memory cost during discriminator backward pass. | Wang et al., *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs* (pix2pixHD, CVPR 2018). |

### Recommended Approach
*(To be filled during implementation)*

### Research Notes
*(Empty — to be populated during implementation)*

---

## Step 4: Implement Generator + Discriminator

### Scope
Build the modular PyTorch neural network classes for the U-Net Generator and PatchGAN Discriminator with integrated style conditioning, alongside paired spatial data transformations.

### What to Build
1. **Paired Augmentation & Dataset Loading** (`src/task4/augmentation.py`):
   - Coupled transformation pipeline applying identical random spatial operations (horizontal flip with $p=0.5$, random crop with padding, affine rotation within $\pm 10^\circ$) to both photo $x$ and sketch $y$.
   - Photo-only photometric variations (subtle brightness/contrast adjustment) that do not alter target sketch line intensity.
   - `FS2KDataset` class handling paired image loading, 128×128 normalization to $[-1, 1]$, and train/val/test split parsing with style stratification.
2. **Conditional U-Net Generator** (`src/task4/generator.py`):
   - Style embedding layer mapping style integer $s \in \{0, 1, 2\}$ to dense representation $e_s \in \mathbb{R}^{d_s}$.
   - Encoder: 4–5 downsampling blocks (`Conv2d` + `InstanceNorm2d` + `LeakyReLU(0.2)`).
   - Bottleneck with dropout ($p \in [0.0, 0.5]$) and style conditioning injection (per Steps 1 & 2).
   - Decoder: 4–5 upsampling blocks (`ConvTranspose2d` or `Upsample` + `Conv2d` with skip connection concatenation + `InstanceNorm2d` + `ReLU`).
   - Output projection: `Conv2d` to 3 channels (or 1 channel) with `Tanh` activation mapping output to $[-1, 1]$.
3. **Conditional PatchGAN Discriminator** (`src/task4/discriminator.py`):
   - Receives concatenated tensor of photo $x$, sketch candidate $y$ (real or fake), and spatially replicated style condition $e_s$.
   - Convolutional layers with `LeakyReLU(0.2)` and `InstanceNorm2d`.
   - Output layer yielding 2D patch logits (without sigmoid) for numerical stability with `BCEWithLogitsLoss`.

### Key Details
- **Normalization**: `InstanceNorm2d` without running statistics (standard for image-to-image translation with small batch sizes).
- **Weight Initialization**: Gaussian initialization $\mathcal{N}(0, 0.02)$ for convolutional and normalization layers.
- **Modularity & Size**: Single-responsibility files with strict length limits under 300 lines of code.

### Verification
- Unit test passing dummy tensors:
  - Input photo: `[B, 3, 128, 128]`, Style index: `[B]`.
  - Generator output shape verified: `[B, 3, 128, 128]` (or `[B, 1, 128, 128]`).
  - Discriminator output shape verified: `[B, 1, H_p, W_p]` logits.
- Unit test verifying identical transformation seed applies matching geometric distortion to both photo and sketch.

### Files Changed / Created
- `src/task4/generator.py`
- `src/task4/discriminator.py`
- `src/task4/augmentation.py`
- `tests/test_task4_models.py`

---

## Step 5: GAN Training Loop

### Scope
Implement the alternating min-max conditional GAN training engine with decoupled loss logging, validation tracking, and fixed-sample visual reconstruction grids.

### What to Build
1. **Alternating Optimization Engine** (`src/task4/train.py`):
   - **Step 1 (Train Discriminator $D$)**:
     - Forward real pair $(x, y, s) \rightarrow D(x, y, s)$.
     - Compute real loss $\mathcal{L}_{D,\text{real}} = \text{BCEWithLogits}(D(x, y, s), \mathbf{1})$ (with optional one-sided label smoothing e.g. 0.9).
     - Generate fake sketch $\hat{y} = G(x, s)$.detach().
     - Forward fake pair $(x, \hat{y}, s) \rightarrow D(x, \hat{y}, s)$.
     - Compute fake loss $\mathcal{L}_{D,\text{fake}} = \text{BCEWithLogits}(D(x, \hat{y}, s), \mathbf{0})$.
     - Total D loss: $\mathcal{L}_D = \frac{1}{2} (\mathcal{L}_{D,\text{real}} + \mathcal{L}_{D,\text{fake}})$.
     - Backpropagate and update $D$ optimizer.
   - **Step 2 (Train Generator $G$)**:
     - Generate fake sketch $\hat{y} = G(x, s)$ (preserving computation graph).
     - Adversarial loss: $\mathcal{L}_{\text{adv}} = \text{BCEWithLogits}(D(x, \hat{y}, s), \mathbf{1})$.
     - L1 reconstruction loss: $\mathcal{L}_{L1} = \| y - \hat{y} \|_1$.
     - Total G loss: $\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \lambda_{L1} \cdot \mathcal{L}_{L1}$.
     - Backpropagate and update $G$ optimizer.
2. **Decoupled Metric & Loss Logging**:
   - Log separately per epoch: `d_real_loss`, `d_fake_loss`, `total_d_loss`, `g_adv_loss`, `g_l1_loss`, `total_g_loss`.
   - Compute validation metrics per epoch: Val L1 error, Val PSNR, and Val SSIM.
3. **Fixed-Sample Visual Progress Grid**:
   - Reserve 6 fixed validation face photos (2 per style category).
   - At fixed epoch intervals (default every 5 epochs), generate sketches $\hat{y}$ using the fixed photos and log a side-by-side visual comparison grid `[Photo | Ground Truth Sketch | Generated Sketch]` to track stylistic and structural evolution.
4. **Checkpoint Management**:
   - Save rolling latest checkpoint and best generator checkpoint based on validation reconstruction metric.

### Key Details
- **Optimizers**: Adam with $\beta_1 = 0.5, \beta_2 = 0.999$, initial learning rate $2\text{e-}4$.
- **Mixed Precision**: Supported via `torch.amp.autocast` and separate `GradScaler` instances for $G$ and $D$.
- **Device Handling**: Uniformly routed via `src.shared.config.get_device()`.

### Verification
- Single-epoch dry run verifying that gradients update both networks without explosion or zero gradients.
- Confirm logging hooks emit all six loss components and image grids to tracking backend.

### Files Changed / Created
- `src/task4/train.py`

---

## Step 6: Training & Validation Run

### Scope
Execute full baseline training run (~100–200 epochs) using standard assignment baseline parameters ($\lambda_{L1} = 100$, default learning rates), analyze convergence stability, and benchmark per-style validation performance.

### What to Build / Run
1. **Execution**:
   - Train on FS2K training split (1,788 pairs), validating every epoch on the 15% stratified validation set (316 pairs).
   - Training schedule: 150 epochs (constant learning rate for first 75 epochs, linear decay to 0 over remaining 75 epochs).
2. **Monitoring & Diagnostic Checks**:
   - Verify non-divergence: $D$ loss remains balanced in range $[0.3, 0.7]$; $G_{\text{adv}}$ loss remains bounded.
   - Track $G_{L1}$ loss progression to confirm progressive facial contour and landmark refinement.
3. **Per-Style Breakdown**:
   - Compute validation L1, SSIM, and PSNR independently across Style 1, Style 2, and Style 3 to detect style-specific convergence variance.
4. **Artifact Storage**:
   - Save baseline checkpoint to `checkpoints/task4/baseline_generator.pth`.
   - Store visual progression grid across epochs 1, 25, 50, 100, 150.

### Verification
- **Stability Criterion**: Neither $D$ loss collapsing to zero nor $G$ diverging to NaN.
- **Qualitative Criterion**: Generated sketches on validation set exhibit recognizable facial identities with distinct stroke textures matching the conditioning style.

### Files Changed / Created
- `checkpoints/task4/baseline_generator.pth`
- Experiment tracking run: `genai-task4-baseline`

---

## Step 7: Optuna Search — GAN Hyperparameters

### Scope & Strategy
Execute Bayesian hyperparameter optimization targeting generator reconstruction fidelity and training stability. Because full adversarial training is computationally expensive, trials run for a reduced epoch budget (35–45 epochs per trial) with automated pruning of unpromising runs.

### Optuna Configuration
- **Study Name**: `task4-cgan`
- **Storage**: SQLite backend at `optuna/optuna_studies.db`
- **Objective**: Minimize Validation L1 Loss (`val_l1_loss`).
  *(Note: Val L1 is chosen over FID/LPIPS for optimization because it is computationally fast to evaluate per epoch while remaining strongly correlated with facial identity retention and geometry alignment)*.
- **Direction**: `minimize`
- **Pruner**: `MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)`
- **Number of Trials**: 15–25 trials
- **Trial Epochs**: 35–45 epochs per trial

### Search Space

| Hyperparameter | Type | Range / Choices | Distribution | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `g_lr` | Float | $[1\text{e-}4, 2\text{e-}3]$ | Log-uniform | Generator step size; crucial for adversarial balance. |
| `d_lr` | Float | $[1\text{e-}4, 2\text{e-}3]$ | Log-uniform | Discriminator step size (can use Two Time-scale Update Rule TTUR). |
| `batch_size` | Categorical | $\{4, 8, 16\}$ | Categorical | Tradeoff between gradient noise, memory, and batch statistics. |
| `base_channels` | Categorical | $\{32, 64\}$ | Categorical | Model capacity vs. parameter footprint and inference latency. |
| `dropout` | Float | $[0.0, 0.5]$ | Uniform | Regularization in generator decoder/bottleneck. |
| `style_embed_dim` | Categorical | $\{8, 16, 32\}$ | Categorical | Latent capacity of the categorical style representation. |
| `lambda_l1` | Float | $[10.0, 200.0]$ | Log-uniform | Tradeoff between edge crispness (adv) and structural accuracy (L1). |

### Key Details
- **Intermediate Reporting**: Report `val_l1_loss` every epoch via `trial.report(val_l1, step=epoch)`.
- **Exception Handling**: Catch divergence or NaNs, report `float('inf')`, and trigger pruning.
- **Logging**: Mirror trial parameters and best validation scores to tracking system (`genai-task4-optuna`).

### Verification
- Study completes designated trials without SQLite database locking errors.
- Best parameter configuration identified and exported as JSON artifact.

### Files Changed / Created
- `src/task4/optuna_search.py`
- `optuna/optuna_studies.db`
- `results/task4/optuna_best_params.json`

---

## Step 8: Final Retrain with Best Config

### Scope
Retrain the conditional GAN model from scratch on the full training schedule (~150–200 epochs) utilizing the optimal hyperparameter tuple identified by Optuna in Step 7.

### What to Build / Run
1. **Model Instantiation**:
   - Construct Generator and Discriminator using optimal `base_channels`, `dropout`, and `style_embed_dim`.
2. **Training Execution**:
   - Train with optimal `g_lr`, `d_lr`, `batch_size`, and $\lambda_{L1}$.
   - Linear learning rate decay schedule over the second half of training.
   - Full evaluation on validation split at every epoch.
3. **Artifact Production**:
   - Save final optimal generator weights: `checkpoints/task4/best_generator.pth`.
   - Log run as `genai-task4-final` in experiment tracking system.

### Verification
- Validation L1, SSIM, and PSNR meet or exceed Step 6 baseline metrics.
- Visual inspection confirms sharp sketch stroke boundaries and high facial landmark fidelity.

### Files Changed / Created
- `checkpoints/task4/best_generator.pth`
- `checkpoints/task4/best_discriminator.pth` (training checkpoint)

---

## Step 9: Evaluation & Visual Results

### Scope
Conduct rigorous quantitative and qualitative evaluation on the held-out FS2K test set. Generate multi-style visual grids, evaluate per-style metrics, and perform diagnostic failure case analysis.

### What to Build
1. **Evaluation Pipeline** (`src/task4/evaluate.py`):
   - Compute metrics across test set:
     - **L1 Error (MAE)**: Pixel-level reconstruction fidelity.
     - **SSIM**: Structural similarity of sketch contours.
     - **LPIPS**: Perceptual distance using pretrained AlexNet/VGG network.
     - **FID (Fréchet Inception Distance)**: Distributional realism of generated sketches.
   - Aggregate metrics reported overall and partitioned per style:
     - Style 1 performance breakdown.
     - Style 2 performance breakdown.
     - Style 3 performance breakdown.
2. **Visual Result Artifacts** (`results/task4/`):
   - **Diverse Sample Grid**: 12+ paired test examples spanning gender, age, facial orientation, and accessories across all 3 styles.
   - **Multi-Style Conditioning Grid**: A fixed panel of identical input test photos synthesized across Style 1, Style 2, and Style 3 side-by-side with ground-truth sketches, demonstrating effective stylistic differentiation under fixed identity.
   - **Failure Analysis Panel**: 4+ failure cases (e.g., severe self-occlusion, heavy glasses reflection, unusual hairstyle texture) with analytical commentary explaining generator degradation modes.

### Key Details
- **Metric Export**: Write structured test metric summary table to `results/task4/test_metrics.json` and markdown report.
- **Visual Figure Generation**: High-DPI comparison figures formatted for IEEE report inclusion.

### Verification
- Evaluation script runs end-to-end on test partition without memory exhaustion.
- Multi-style comparison clearly reflects distinct artistic attributes per style index.

### Files Changed / Created
- `src/task4/evaluate.py`
- `results/task4/test_metrics.json`
- `results/task4/sample_results_grid.png`
- `results/task4/style_comparison_grid.png`
- `results/task4/failure_cases_analysis.png`

---

## Step 10: ONNX Export & Verification

### Scope
Export the trained generator network to an optimized ONNX computational graph (opset $\ge 17$) for low-latency CPU/GPU deployment in the FastAPI application workspace, followed by rigorous numerical parity verification against native PyTorch.

### What to Build
1. **Export Script** (`src/task4/export_onnx.py`):
   - Load `checkpoints/task4/best_generator.pth` into generator $G$.
   - Set $G$ to evaluation mode (`model.eval()`).
   - Define model input signatures:
     - `photo`: Float32 tensor of shape `(B, 3, 128, 128)`.
     - `style_index`: Int64 tensor of shape `(B,)` containing categorical style indices $\in \{0, 1, 2\}$.
   - Target output:
     - `sketch`: Float32 tensor of shape `(B, 3, 128, 128)` (or `(B, 1, 128, 128)`).
   - Export parameters:
     - `opset_version = 17`
     - Dynamic axes configuration:
       ```python
       dynamic_axes = {
           "photo": {0: "batch_size"},
           "style_index": {0: "batch_size"},
           "sketch": {0: "batch_size"}
       }
       ```
     - Output model destination: `models/onnx/task4_generator.onnx`.
2. **Numerical Parity Verification**:
   - Execute inference on identical validation test batch ($B=8$) using:
     1. Native PyTorch model ($y_{\text{pt}} = G(x, s)$).
     2. ONNX Runtime session ($y_{\text{ort}} = \text{session.run}(\dots)$).
   - Verify numerical agreement:
     $$\max | y_{\text{pt}} - y_{\text{ort}} | < 10^{-5}$$
     asserting `np.allclose(y_pt, y_ort, atol=1e-5)`.
3. **Application Workspace Integration Contract**:
   - Endpoint: `/api/v1/sketch/generate`
   - Flow:
     - Client uploads photo (file upload or webcam capture) and selects style (1, 2, or 3).
     - Backend resizes image to $128 \times 128$, normalizes tensor, and queries ONNX Runtime engine with `photo` and `style_index`.
     - Synthesized sketch is returned as base64 or binary image for side-by-side UI rendering and user download.

### Key Details
- Discriminator is strictly for training and is **not** exported.
- Style embedding table is embedded directly inside the ONNX graph; the runtime input only requires the integer style index.
- Run ONNX model validation using `onnx.checker.check_model`.

### Verification
- `export_onnx.py` completes without graph conversion warnings.
- `np.allclose(y_pt, y_ort, atol=1e-5)` assertion succeeds.
- Inference latency benchmarked and logged.

### Files Changed / Created
- `src/task4/export_onnx.py`
- `models/onnx/task4_generator.onnx`
- `tests/test_task4_onnx.py`
