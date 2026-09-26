# Task 3 Plan: Soft Mixture-of-Experts (MoE) Image Restoration

## Overview & Architecture Context

This document outlines the step-by-step engineering plan for **Task 3** of the Generative AI Assignment: transforming the discrete hard-routing restoration system (Task 2) into a differentiable **Soft Mixture-of-Experts (MoE)** restoration model.

### Mathematical Formulation

Given a corrupted input image $\tilde{x} \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ from the Oxford-IIIT Pet dataset:
1. **Gating Network ($G$):** Evaluates the corrupted input and outputs routing logits $z = G(\tilde{x}) \in \mathbb{R}^{B \times 4}$.
2. **Temperature-Scaled Routing Weights ($w$):**
   $$w = \text{softmax}\left(\frac{G(\tilde{x})}{\tau}\right) = [w_1, w_2, w_3, w_4]$$
   where $\sum_{k=1}^4 w_k = 1$, and $\tau > 0$ controls routing entropy (sharp vs. distributed blending).
   - $w_1$: Clean / Identity branch weight
   - $w_2$: Salt-and-Pepper specialist weight ($A_{\text{salt}}$)
   - $w_3$: Gaussian Blur specialist weight ($A_{\text{blur}}$)
   - $w_4$: Rectangular Occlusion specialist weight ($A_{\text{occlusion}}$)
3. **Composite Restoration ($\hat{x}$):**
   $$\hat{x} = w_1 \cdot \tilde{x} + w_2 \cdot A_{\text{salt}}(\tilde{x}) + w_3 \cdot A_{\text{blur}}(\tilde{x}) + w_4 \cdot A_{\text{occlusion}}(\tilde{x})$$
4. **Joint Multi-Objective Loss ($\mathcal{L}_{\text{tot}}$):**
   $$\mathcal{L}_{\text{tot}} = \lambda_1 \mathcal{L}_1(\hat{x}, x) + \lambda_2 (1 - \text{SSIM}(\hat{x}, x)) + \lambda_3 \mathcal{L}_{\text{CE}}(G(\tilde{x}), y) + \lambda_4 \mathcal{L}_{\text{balance}}(w)$$
   - Ground truth target: clean image $x \in \mathbb{R}^{B \times 3 \times 128 \times 128}$
   - Corruption class label: $y \in \{0, 1, 2, 3\}$
   - Initial hyperparameters: $\lambda_1 = 0.8$, $\lambda_2 = 0.2$, $\lambda_3 = 0.1$, $\lambda_4 = 0.01$

---

## Step 1: Gating Network & MoE Architecture Research

### Decision & Why It Matters
The gating network $G(\tilde{x})$ maps the input image to a 4-dimensional routing distribution across experts. This decision determines:
1. Transferability of features from the pre-trained Task 2 corruption classifier (`checkpoints/task2/classifier_best.pth`).
2. Expressive capacity to detect composite, borderline, or low-severity corruptions.
3. Computational overhead during inference (since all 4 branches must execute during forward propagation).
4. Gradient flow behavior through the temperature-scaled softmax layer.

### Alternatives to Research

| Dimension | Option 1: Linear Gate | Option 2: Multi-layer Perceptron (MLP) Gate | Option 3: Attention-Based Feature Gate |
| :--- | :--- | :--- | :--- |
| **Architecture** | Feature backbone from Task 2 classifier + Global Average Pooling (GAP) + Single Linear layer ($D \to 4$). | Feature backbone + GAP + 2 FC layers with ReLU/BatchNorm ($D \to 128 \to 4$). | Conv backbone + Squeeze-and-Excitation / Spatial-Channel cross-attention pooling + Linear projection. |
| **Parameter Count** | $\approx 2.05\text{K}$ gate params (assuming $D=512$). | $\approx 66\text{K}$ gate params. | $\approx 280\text{K}$ gate params. |
| **Init from Task 2 Classifier** | Direct 1:1 weight transfer of both convolutional backbone and linear head. | Direct transfer of backbone; head requires surgical adaptation or random initialization. | Backbone transfer requires adding attention modules; cannot directly initialize linear head. |
| **Routing Capacity** | Linear decision boundaries in pooled feature space; optimal for distinct corruptions. | Non-linear decision boundaries; handles complex feature intersections. | High capacity; models spatially varying corruption patterns. |
| **Routing Sharpness Control** | Strictly dictated by temperature $\tau$ and logit scaling. | Governed by $\tau$ and non-linear layer activations. | Governed by $\tau$ and internal attention temperature. |
| **Inference Overhead** | Minimal ($< 0.5\text{ ms}$). | Low ($< 1.0\text{ ms}$). | Moderate ($3\text{--}5\text{ ms}$). |

### Recommended Approach
*To be filled during implementation based on Task 2 classifier backbone compatibility and empirical routing stability.*

### Research Notes
*(Empty section — to be populated with experimental observations during implementation).*

---

## Step 2: Implement Soft MoE Model

### Scope
Construct the unified `SoftMoE` PyTorch module encapsulating the Gating Network, Identity branch, and three pre-trained Specialist Autoencoders. Implement the temperature-scaled softmax forward pass and convex output blending.

### What to Build
1. `src/task3/gate.py`:
   - `GatingNetwork`: Wraps the convolutional feature extractor, global average pooling, and linear classification head.
   - Forward pass accepts input $\tilde{x}$ and optional temperature scalar $\tau$, returning unnormalized logits $z$ and normalized routing weights $w = \text{softmax}(z / \tau, \dim=-1)$.
   - Clamping safeguards on $\tau$ ($\tau \ge 0.05$) to prevent numerical overflow or division-by-zero.
2. `src/task3/moe_model.py`:
   - `SoftMoE`: Top-level model holding:
     - `self.gate`: Instance of `GatingNetwork`.
     - `self.salt_expert`: Specialist autoencoder $A_{\text{salt}}$.
     - `self.blur_expert`: Specialist autoencoder $A_{\text{blur}}$.
     - `self.occlusion_expert`: Specialist autoencoder $A_{\text{occlusion}}$.
   - `forward(x, tau=1.0)` method:
     1. Compute $w = \text{softmax}(G(x) / \tau)$.
     2. Evaluate branches:
        - $b_1 = x$ (Identity / Clean branch)
        - $b_2 = A_{\text{salt}}(x)$
        - $b_3 = A_{\text{blur}}(x)$
        - $b_4 = A_{\text{occlusion}}(x)$
     3. Compute linear combination:
        $$\hat{x} = \sum_{k=1}^4 w_{:, k, \text{None}, \text{None}, \text{None}} \odot b_k$$
     4. Return tuple `(reconstruction, routing_weights, logits)`.

### Key Details
- **Identity Branch:** Has zero trainable parameters and performs no transformations ($b_1 = x$).
- **Differentiability:** Weight broadcasting preserves computation graph from output $\hat{x}$ back to both the expert weights and the gating network parameters.
- **Memory Optimization:** Model must support batch processing on 128×128 RGB images without memory leaks.

### Verification
- Unit test passing dummy tensor `(B, 3, 128, 128)` through `SoftMoE`:
  - Assert reconstruction shape equals `(B, 3, 128, 128)`.
  - Assert routing weights shape equals `(B, 4)`.
  - Assert $\sum_{k=1}^4 w_{i,k} = 1.0 \pm 10^{-6}$ across all batch indices $i$.
- Backward pass gradient verification: non-zero gradients verified across all active expert modules and gate parameters.

### Files Changed
- `src/task3/gate.py`
- `src/task3/moe_model.py`

---

## Step 3: Training Strategy — Warm-up → Joint Fine-tune

### Scope
Implement the two-phase training protocol to stabilize gate alignment and prevent specialist degradation:
- **Phase 1 (Warm-up):** Freeze all specialist autoencoder weights ($\theta_{\text{experts}}$). Initialize gating network from Task 2 classifier checkpoint. Train only the gate parameters using joint loss $\mathcal{L}_{\text{tot}}$ for $N_{\text{warmup}}$ epochs.
- **Phase 2 (Joint Fine-tuning):** Unfreeze all specialist autoencoders. Lower the learning rate ($\eta_{\text{joint}} < \eta_{\text{warmup}}$) with differential parameter groups. Fine-tune the entire MoE end-to-end for $N_{\text{joint}}$ epochs.

### What to Build
- `src/task3/train.py`:
  - `load_pretrained_components()`: Utility to map weights from `checkpoints/task2/classifier_best.pth` and specialist checkpoints (`checkpoints/task2/specialist_salt_best.pth`, etc.) into `SoftMoE`.
  - `set_experts_frozen(model, frozen=True)`: Utility toggling `requires_grad` for all expert weights.
  - Multi-objective loss calculator:
    $$\mathcal{L}_{\text{tot}} = \lambda_1 \mathcal{L}_1(\hat{x}, x) + \lambda_2 (1 - \text{SSIM}(\hat{x}, x)) + \lambda_3 \mathcal{L}_{\text{CE}}(z, y) + \lambda_4 \mathcal{L}_{\text{balance}}(w)$$
  - Two-stage training orchestrator handling optimizer re-initialization and learning rate scheduling across transition boundaries.

### Key Details
- **Warm-up Hyperparameters:** $N_{\text{warmup}} = 5\text{--}10$ epochs, Adam optimizer with $\text{LR} = 1 \times 10^{-3}$, cosine annealing.
- **Joint Hyperparameters:** $N_{\text{joint}} = 25\text{--}40$ epochs, Adam optimizer with differential learning rates:
  - Gate LR: $1 \times 10^{-4}$
  - Specialist AE LR: $2 \times 10^{-5}$
- **Auxiliary CE Loss:** Uses ground-truth corruption label $y \in \{0, 1, 2, 3\}$ to reinforce semantic alignment of the routing logits during warm-up.

### Verification
- Phase 1 validation: Gate classification accuracy and routing alignment improve steadily while expert weights remain bit-exact identical to Task 2 checkpoints.
- Phase 2 validation: Joint fine-tuning produces strictly lower combined reconstruction error $\lambda_1 \mathcal{L}_1 + \lambda_2 (1 - \text{SSIM})$ compared to the end of Phase 1.

### Files Changed
- `src/task3/train.py`

---

## Step 4: Balance Regularizer Research & Selection

### Decision & Why It Matters
In Mixture-of-Experts architectures, gating networks frequently suffer from **routing collapse** (expert starvation), where one or two specialists dominate all inputs while others receive zero gradient updates. The balance regularizer $\mathcal{L}_{\text{balance}}(w)$ penalizes degenerate distributions and enforces expert utilization without overpowering specialized task partitioning.

### Alternatives to Research

| Dimension | Option 1: L2 Deviation from Uniformity | Option 2: Batch-Mean Entropy Regularizer | Option 3: Switch Transformer Load Balancing Loss |
| :--- | :--- | :--- | :--- |
| **Formula** | $\mathcal{L}_{\text{balance}} = \sum_{k=1}^4 \left(\bar{w}_k - \frac{1}{4}\right)^2$<br>where $\bar{w}_k = \frac{1}{B}\sum_{i=1}^B w_{i,k}$ | $\mathcal{L}_{\text{balance}} = \sum_{k=1}^4 \bar{w}_k \log(\bar{w}_k + \epsilon)$<br>(negative entropy of batch mean) | $\mathcal{L}_{\text{balance}} = 4 \sum_{k=1}^4 f_k \cdot \bar{w}_k$<br>where $f_k = \frac{1}{B}\sum_{i=1}^B \mathbb{I}\{\text{argmax}(w_i) = k\}$ |
| **Target Distribution** | Exactly uniform batch distribution ($0.25$ per expert). | High entropy batch distribution; smooth gradients across probabilities. | Differentiable proxy for hard routing assignment fraction $\times$ soft probability. |
| **Gradient Dynamics** | Linear penalty gradient proportional to difference from $0.25$. | Steeper gradients when an expert approaches starvation ($\bar{w}_k \to 0$). | Coupled interaction between assignment frequency and routing mass. |
| **Risk of Over-smoothing** | Low; allows high single-sample confidence as long as batch is balanced. | Moderate; high weight $\lambda_4$ can force uniform routing for individual samples. | Low; extensively validated in large-scale MoE language and vision models. |
| **Assignment Alignment** | Directly suggested in assignment specification. | Standard information-theoretic alternative. | State-of-the-art literature baseline (Fedus et al., 2021). |

### Side-by-Side Experiment Protocol
Execute 10-epoch runs with identical seed (`42`), identical initial learning rate ($1 \times 10^{-4}$), and fixed initial regularizer coefficient ($\lambda_4 = 0.01$). Evaluate:
1. Validation PSNR and SSIM.
2. Routing distribution vector $\bar{w} = [\bar{w}_{\text{clean}}, \bar{w}_{\text{salt}}, \bar{w}_{\text{blur}}, \bar{w}_{\text{occ}}]$.
3. Minimum expert utilization across all validation batches.

#### Comparison Results

| Regularizer Variant | Val PSNR (dB) | Val SSIM | Val L1 Loss | Avg Routing Vector $[\bar{w}_1, \bar{w}_2, \bar{w}_3, \bar{w}_4]$ | Collapse Detected? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2 Deviation** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Entropy Maximization** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Switch Load Balance** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### Recommended Approach
*To be filled during implementation based on side-by-side empirical results.*

### Research Notes
*(Empty section — to be populated with experimental observations during implementation).*

---

## Step 5: Training & Validation Run

### Scope
Execute the full baseline training schedule utilizing the warm-up → joint fine-tune strategy and selected balance regularizer under initial assignment weighting:
$$\lambda_1 = 0.8, \quad \lambda_2 = 0.2, \quad \lambda_3 = 0.1, \quad \lambda_4 = 0.01$$

### What to Build
- Logging harness integration with MLflow/W&B:
  - Track loss components: $\mathcal{L}_{\text{total}}$, $\mathcal{L}_1$, $1-\text{SSIM}$, $\mathcal{L}_{\text{CE}}$, $\mathcal{L}_{\text{balance}}$.
  - Track image quality metrics: Validation PSNR, SSIM, L1.
  - Track per-corruption mean routing vectors: $\bar{w}_{\text{clean}}, \bar{w}_{\text{salt}}, \bar{w}_{\text{blur}}, \bar{w}_{\text{occ}}$.
- Checkpointing logic:
  - Save `checkpoints/task3/baseline_best.pth` on validation SSIM improvement.
  - Save `checkpoints/task3/latest.pth` every epoch.

### Key Details
- Dataset: Oxford-IIIT Pet 80% train / 20% validation (`random_seed=42`).
- Fixed deterministic validation manifest (`manifests/val_corruptions.json`) ensuring unbiased evaluation across epochs.
- Early detection of routing anomalies: alert if any expert achieves $> 0.90$ average weight across the entire validation set.

### Verification
- Training loss steadily converges without oscillations or numeric instability.
- Validation PSNR exceeds Task 1 Universal baseline ($> 24.5\text{ dB}$).
- Checkpoints cleanly serialize and deserialize model state dict, optimizer state, epoch, and metric history.

### Files Changed
- `checkpoints/task3/baseline_best.pth`
- `checkpoints/task3/latest.pth`
- `results/task3/baseline_training_log.json`

---

## Step 6: Optuna Search — Joint Fine-tuning

### Study Definition
- **Study Name:** `task3-moe-joint`
- **Storage:** SQLite database at `optuna/optuna_studies.db`
- **Objective:** Maximize Validation SSIM (or minimize combined validation reconstruction loss)
- **Direction:** `maximize`
- **Number of Trials:** 25 trials
- **Pruner:** `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)` combined with a custom **Routing Collapse Pruning Callback**.

### Search Space

| Hyperparameter | Distribution | Range | Description |
| :--- | :--- | :--- | :--- |
| `fine_tune_lr` | Float (log) | $[1 \times 10^{-5}, 1 \times 10^{-3}]$ | Learning rate during joint fine-tuning phase. |
| `temperature` ($\tau$) | Float (linear) | $[0.1, 5.0]$ | Softmax temperature governing sharpness vs. distribution of routing. |
| `lambda_ce` ($\lambda_3$) | Float (log) | $[0.01, 0.5]$ | Weight of cross-entropy auxiliary loss on gate routing. |
| `lambda_balance` ($\lambda_4$) | Float (log) | $[0.001, 0.1]$ | Weight of balance loss preventing expert starvation. |
| `reconstruction_alpha` ($\alpha$) | Float (linear) | $[0.5, 1.0]$ | Relative tradeoff between L1 ($\alpha$) and SSIM ($1-\alpha$) in reconstruction. |

### Custom Routing Collapse Pruning Rule
At the end of each validation epoch within a trial:
1. Compute the global average routing weight for each branch across the validation set: $\bar{w}_k = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} w_{i,k}$.
2. If $\max_{k \in \{1,2,3,4\}} \bar{w}_k > 0.90$ or $\min_{k \in \{1,2,3,4\}} \bar{w}_k < 0.02$:
   - Trigger immediate pruning: `raise optuna.TrialPruned("Routing collapse detected: expert dominance > 90% or starvation < 2%")`.

### Verification
- Optuna database successfully records all trial parameters, intermediate epoch reports, and pruning events.
- Best trial parameters exported to `config/task3_best_params.json`.
- Visualizations generated: optimization history plot, hyperparameter slice plot, and parameter importances plot.

### Files Changed
- `src/task3/optuna_search.py`
- `optuna/optuna_studies.db`
- `config/task3_best_params.json`

---

## Step 7: Final Retrain with Best Config

### Scope
Execute full training run using the optimal hyperparameter configuration discovered in Step 6. Train from Task 2 pre-trained weights through complete warm-up and joint fine-tuning schedules.

### What to Build
- Script `src/task3/train_final.py` (or flags in `train.py`) loading `config/task3_best_params.json`.
- Extended training schedule: 10 epochs warm-up + 40 epochs joint fine-tune.
- Comprehensive checkpointing saving full metadata: model weights, optimizer, epoch, metrics, and hyperparameter configuration.
- Tag run as `final` in MLflow/W&B.

### Key Details
- Seed: Explicitly set `random_seed=42` across PyTorch, NumPy, and CUDA.
- Final model weights saved to `checkpoints/task3/best_model.pth`.
- Produce per-corruption, per-severity validation metrics at the conclusion of training.

### Verification
- Validation SSIM and PSNR meet or exceed the performance of the best Optuna trial.
- Absence of routing collapse confirmed across all evaluation splits.

### Files Changed
- `checkpoints/task3/best_model.pth`
- `results/task3/final_val_metrics.json`

---

## Step 8: Routing Analysis & Visualization

### Scope
Perform exhaustive analysis of the gating network's routing behavior across all corruption types and severities to produce required IEEE report visualizations and diagnostic tables.

### What to Build
- `src/task3/routing_analysis.py`: Standalone analysis script that loads `checkpoints/task3/best_model.pth` and computes:
  1. **4×4 Routing Weight Confusion Matrix:**
     Average expert routing weight vector $[w_1, w_2, w_3, w_4]$ conditioned on each ground-truth corruption category:
     - True Clean $\to [w_{\text{clean}}, w_{\text{salt}}, w_{\text{blur}}, w_{\text{occ}}]$
     - True Salt-and-Pepper $\to [w_{\text{clean}}, w_{\text{salt}}, w_{\text{blur}}, w_{\text{occ}}]$
     - True Gaussian Blur $\to [w_{\text{clean}}, w_{\text{salt}}, w_{\text{blur}}, w_{\text{occ}}]$
     - True Rectangular Occlusion $\to [w_{\text{clean}}, w_{\text{salt}}, w_{\text{blur}}, w_{\text{occ}}]$
  2. **Severity-Dependent Routing Trends:**
     Curves showing routing weight progression as corruption severity escalates:
     - Salt-and-pepper noise density $p \in \{0.03, 0.08, 0.15\}$ vs. $w_2$
     - Gaussian blur $(\text{kernel}, \sigma) \in \{(3,0.7), (5,1.5), (7,2.5)\}$ vs. $w_3$
     - Rectangular occlusion area $\in \{10\%, 20\%, 35\%\}$ vs. $w_4$
  3. **Routing Heatmap Matrix:** Visual matrix visualization plotting input instances/classes against expert activation weights.
  4. **Sharp Routing Gallery:** Sample images where one expert exhibits extreme dominance ($w_k > 0.85$).
  5. **Soft Routing Gallery:** Sample images where multiple experts receive balanced non-zero allocations (e.g., subtle corruptions or boundary cases).
  6. **Expert Health Checks:**
     - Inactive Expert Check: Verify no specialist has average dataset weight $< 5\%$.
     - Unrelated Dominance Check: Verify specialist $k$ does not dominate unrelated corruption inputs (e.g., blur expert dominating salt noise).

### Verification
- All plots rendered and saved in high-resolution PDF and PNG formats to `results/task3/figures/`.
- Summary tables exported to CSV for direct inclusion in the IEEE LaTeX document.

### Files Changed
- `src/task3/routing_analysis.py`
- `results/task3/routing_confusion_matrix.csv`
- `results/task3/figures/routing_heatmap.png`
- `results/task3/figures/severity_routing_trends.png`
- `results/task3/figures/sharp_vs_distributed_examples.png`

---

## Step 9: Evaluation & Comparison with Task 1/2

### Scope
Conduct rigorous final test-set evaluation using the untouched official test split of Oxford-IIIT Pet. Compare Task 3 Soft MoE directly against Task 1 Universal Autoencoder, Task 2 Oracle-Routed Specialists, and Task 2 Predicted-Routed Specialists.

### What to Build
- `src/task3/evaluate.py`:
  - Loads test manifests (`manifests/test_corruptions.json`).
  - Computes PSNR, SSIM, and L1 error across every corruption type and severity level.
  - Aggregates comparative performance into a unified benchmark table.
- Quantitative Win/Loss Analysis:
  - Benchmark performance across all 4 systems.
  - Identification of regimes where Soft MoE outperforms Hard Routing (e.g., misclassification resilience, clean identity preservation).
  - Identification of failure modes or tradeoffs (e.g., ghosting artifacts when two conflicting specialists blend).
- Visual Gallery Generation:
  - 12+ representative qualitative restoration panels across all corruptions and severities.
  - 4+ failure case visualizations with detailed error analysis.
  - Target demonstration of test cases where Task 2 Hard Routing failed due to classifier error, but Task 3 Soft MoE restored successfully due to soft blending.

### Comparison Table Structure

| Method | Overall PSNR | Overall SSIM | Clean PSNR | Salt (Low/Med/High) | Blur (Low/Med/High) | Occ (Low/Med/High) | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Universal AE** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 2.1\text{ ms}$ |
| **Task 2: Hard (Predicted)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 3.2\text{ ms}$ |
| **Task 2: Hard (Oracle)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 2.1\text{ ms}$ |
| **Task 3: Soft MoE** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | $\approx 6.8\text{ ms}$ |

### Verification
- Script generates all metrics deterministically on the official test set.
- Results saved to `results/task3/test_benchmark_comparison.csv`.
- Visual grids saved to `results/task3/figures/qualitative_comparison_12.png` and `results/task3/figures/failure_cases_4.png`.

### Files Changed
- `src/task3/evaluate.py`
- `results/task3/test_benchmark_comparison.csv`
- `results/task3/figures/qualitative_comparison_12.png`
- `results/task3/figures/failure_cases_4.png`

---

## Step 10: ONNX Export & Verification

### Scope
Export the complete end-to-end Soft MoE pipeline to a unified ONNX model for high-performance deployment in FastAPI and the React frontend application workspace.

### Research Sub-Question: Single Combined Graph vs. Decoupled Pipeline
- **Option A (Combined Graph):** Export the full pipeline (Gating Network + Identity pass-through + 3 Specialist branches + dynamic weighted sum) into a single `.onnx` graph.
  - *Pros:* Atomic inference call; minimal serialization/deserialization overhead; clean integration into FastAPI endpoint `/api/v1/restore/soft-moe`.
  - *Cons:* All 4 branches always execute in the computation graph (matches mathematical formulation of soft MoE).
- **Option B (Decoupled Pipeline):** Export 1 Gate ONNX model and 3 Specialist ONNX models; orchestrate the weighted sum in Python/NumPy.
  - *Pros:* Conditional execution possible if gating weights fall below threshold $\epsilon$.
  - *Cons:* Multiple ONNX Runtime sessions, higher latency, complex session management.
- **Recommendation:** **Option A (Combined Graph)**. Aligns with the end-to-end differentiable MoE paradigm and ensures sub-20ms inference in ONNX Runtime.

### Export Specifications
- **Input:** `input_image` $\to$ shape `(B, 3, 128, 128)`, dtype `float32`.
- **Outputs:**
  - `restored_image` $\to$ shape `(B, 3, 128, 128)`, dtype `float32`.
  - `routing_weights` $\to$ shape `(B, 4)`, dtype `float32`.
- **Target Opset:** `opset_version = 17` (ensures full native support for Softmax temperature scaling and 5D tensor broadcasting).
- **Dynamic Axes:**
  ```python
  dynamic_axes = {
      "input_image": {0: "batch_size"},
      "restored_image": {0: "batch_size"},
      "routing_weights": {0: "batch_size"},
  }
  ```

### Verification Procedure
1. Export model via `torch.onnx.export()` to `models/onnx/task3_soft_moe.onnx`.
2. Validate ONNX structural integrity using `onnx.checker.check_model()`.
3. Execute identical validation batch ($B=8$) through both PyTorch eager mode and ONNX Runtime (`CPUExecutionProvider` / `CUDAExecutionProvider`).
4. Validate numerical equivalence:
   $$\max |\hat{x}_{\text{PyTorch}} - \hat{x}_{\text{ORT}}| < 1 \times 10^{-5}$$
   $$\max |w_{\text{PyTorch}} - w_{\text{ORT}}| < 1 \times 10^{-5}$$
   Assert `np.allclose(pytorch_out, ort_out, atol=1e-5)`.
5. Benchmark inference latency across 100 iterations and log speedup metrics for the app workspace (`Soft Mixture-of-Experts Restoration`).

### Files Changed
- `src/task3/export_onnx.py`
- `models/onnx/task3_soft_moe.onnx`
