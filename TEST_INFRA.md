# Task 3 Soft Mixture-of-Experts (MoE) Image Restoration — Test Infrastructure Specification

**Document Version:** 1.0.0  
**Target Milestone:** Task 3 E2E Testing Track (M1–M5 Full System)  
**Primary Test Target:** `tests/test_task3_e2e.py`  
**Execution Command:** `uv run pytest -v tests/test_task3_e2e.py`  

---

## 1. Executive Summary & Test Mission

The mission of the Task 3 E2E test suite is to provide an uncompromising, opaque-box, mathematically rigorous verification harness for the Soft Mixture-of-Experts (MoE) image restoration pipeline. 

Unlike Task 2's discrete, non-differentiable hard routing, Task 3 introduces a continuous, differentiable routing formulation:
$$\hat{x} = \sum_{k=1}^4 w_k \cdot b_k(\tilde{x}), \quad w = \text{softmax}\left(\frac{G(\tilde{x})}{\tau}\right)$$
where $b_1 = \tilde{x}$ (Identity clean bypass), $b_2 = A_{\text{salt}}(\tilde{x})$, $b_3 = A_{\text{blur}}(\tilde{x})$, and $b_4 = A_{\text{occlusion}}(\tilde{x})$.

The test infrastructure validates:
1. Exact probabilistic routing properties ($\sum_{k=1}^4 w_k = 1.0 \pm 10^{-6}$, $w_k \in [0, 1]$).
2. Unbroken end-to-end gradient backpropagation through temperature scaling, softmax, and convex blending into both gating parameters and specialist autoencoders.
3. Strict numerical safeguards against division-by-zero, underflow, and overflow ($\tau = \max(\tau, 0.05)$).
4. Prevention of routing collapse via multi-objective loss formulations and balance regularizers.
5. Bit-exact identity bypass preservation ($w_1 = 1 \implies \hat{x} \equiv \tilde{x}$ with $0.0$ MSE).
6. Atomic single-graph ONNX export (opset 17, dynamic batching) and strict numerical parity ($\max |\hat{x}_{\text{pt}} - \hat{x}_{\text{ort}}| < 10^{-5}$).
7. End-to-end pipeline execution on real Oxford-IIIT Pet dataset images across all 4 corruption modes.

---

## 2. Four-Tier Testing Methodology

The test suite is structured into four complementary testing tiers ensuring exhaustive coverage:

```
+------------------------------------------------------------------------------------+
| Tier 4: Real-World Scenarios                                                       |
|  - E2E Oxford-IIIT Pet dataset loading, corruptions, restoration, PSNR/SSIM audit  |
+------------------------------------------------------------------------------------+
                                      ▲
+------------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Interactions                                                 |
|  - Gate + Specialists + Blending + Multi-objective Loss + Frozen/Unfrozen Dynamics  |
|  - Optuna collapse pruning integration + PyTorch-to-ONNX runtime equivalence       |
+------------------------------------------------------------------------------------+
                                      ▲
+------------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases                                                    |
|  - Temperature limits (tau -> 0, tau -> inf), singleton batch (B=1), prime batch   |
|  - Corrupted extremes (all-black, all-white, NaN/Inf checks, extreme noise p=1.0)  |
+------------------------------------------------------------------------------------+
                                      ▲
+------------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (>=5 test cases per architectural feature M1-M5)          |
|  - Gating Network, Temperature Clamping, SoftMoE Container, Blending, Checkpoints  |
|  - Balance Regularizers, Loss Functions, Pruning Rules, Confusion Matrix, Exporter |
+------------------------------------------------------------------------------------+
```

---

### Tier 1: Feature Coverage (M1–M5)

Every core capability across Milestones 1 through 5 is verified with $\ge 5$ targeted test cases:

#### Feature 1: Gating Network Architecture (`GatingNetwork`)
- **T1.1.1 — Output Topology**: Forward pass on `(B, 3, 128, 128)` yields routing weights shape `(B, 4)` and logits shape `(B, 4)`.
- **T1.1.2 — Softmax Normalization**: Routing weights sum to $1.0 \pm 10^{-6}$ across all rows.
- **T1.1.3 — Probability Range**: Every component $w_{i, k} \in [0.0, 1.0]$ with non-negative lower bound.
- **T1.1.4 — Checkpoint Key Mapping**: Loading from `checkpoints/task2/classifier_best.pt` maps `model.features.*` and `model.fc.*` to gate parameters without missing keys.
- **T1.1.5 — Parameter Count & Capacity**: Verifies gate architecture contains 4 convolutional stages followed by GAP and Linear(256, 4).

#### Feature 2: Temperature Scaling & Numerical Clamping
- **T1.2.1 — Standard Temperature Effect**: $\tau = 1.0$ matches standard softmax $w = \text{softmax}(z)$.
- **T1.2.2 — High Temperature Smoothing**: $\tau = 5.0$ increases entropy, bringing $w$ closer to $[0.25, 0.25, 0.25, 0.25]$.
- **T1.2.3 — Low Temperature Sharpening**: $\tau = 0.1$ concentrates routing mass on the maximum logit.
- **T1.2.4 — Clamp Safeguard Activation**: Passing $\tau < 0.05$ (e.g. $\tau = 0.001$ or $\tau = -1.0$) is clamped to $\tau = 0.05$, preventing NaN/Inf.
- **T1.2.5 — Temperature Invariance on Equal Logits**: Uniform logits $z = [c, c, c, c]$ yield $w = [0.25, 0.25, 0.25, 0.25]$ regardless of $\tau$.

#### Feature 3: SoftMoE Container Module (`SoftMoE`)
- **T1.3.1 — Submodule Composition**: Container initializes with `gate`, `specialist_salt`, `specialist_blur`, `specialist_occlusion`.
- **T1.3.2 — Full Forward Signature**: Forward pass returns 3-tuple `(restored_image, routing_weights, logits)`.
- **T1.3.3 — Output Dimension Fidelity**: Restored image preserves exact spatial shape `(B, 3, 128, 128)`.
- **T1.3.4 — Multi-Batch Scaling**: Evaluates successfully across batch sizes $B \in \{1, 2, 4, 8\}$.
- **T1.3.5 — Eval vs Train Mode**: Model correctly toggles dropout/batchnorm between `model.train()` and `model.eval()`.

#### Feature 4: Differentiable Convex Blending
- **T1.4.1 — Convex Sum Identity**: Blended reconstruction $\hat{x} = \sum_{k=1}^4 w_k b_k$ lies strictly in the convex hull of branch outputs.
- **T1.4.2 — End-to-End Gradient Propagation**: Loss backward passes non-zero gradients into both gate parameters and specialist parameters.
- **T1.4.3 — Weight Sensitivity**: Perturbation of gating logits $\Delta z$ produces direct, non-zero gradient $\frac{\partial \hat{x}}{\partial z}$.
- **T1.4.4 — Orthogonal Branch Contribution**: When routing weights are one-hot $[0, 1, 0, 0]$, $\frac{\partial \hat{x}}{\partial b_3} \equiv 0$ and $\frac{\partial \hat{x}}{\partial b_4} \equiv 0$.
- **T1.4.5 — 5D Tensor Broadcasting**: Weight expansion `w[:, k, None, None, None]` broadcasts correctly without memory reallocation errors.

#### Feature 5: Identity Pass-Through Branch
- **T1.5.1 — Bit-Exact Clean Bypass**: When routing weight $w = [1, 0, 0, 0]$, $\hat{x} \equiv \tilde{x}$ with $\text{MSE}(\hat{x}, \tilde{x}) = 0.0$.
- **T1.5.2 — Zero Trainable Parameters**: Identity branch introduces zero parameters to the model graph.
- **T1.5.3 — Zero Gradient to Branch 1 Weights**: Since branch 1 has no parameters, no spurious gradients are stored.
- **T1.5.4 — Unmodified Dynamic Range**: Clean image in $[0.0, 1.0]$ maintains exact min/max bounds without clipping.
- **T1.5.5 — High-Frequency Preservation**: High-frequency textures in clean images pass through without blurring degradation.

#### Feature 6: Pre-trained Checkpoint Integration & Freezing
- **T1.6.1 — Component Deserialization**: Successfully loads all 4 Task 2 checkpoints (`classifier_best.pt`, `specialist_salt_best.pt`, `specialist_blur_best.pt`, `specialist_occlusion_best.pt`).
- **T1.6.2 — Freeze Toggle Operation**: `set_experts_frozen(True)` sets `requires_grad=False` on all specialist parameters while keeping gate `requires_grad=True`.
- **T1.6.3 — Unfreeze Toggle Operation**: `set_experts_frozen(False)` restores `requires_grad=True` across all specialist parameters.
- **T1.6.4 — Warm-up Parameter Isolation**: Running `loss.backward()` and `optimizer.step()` during warm-up verifies specialist weights remain bit-exact identical.
- **T1.6.5 — Differential Parameter Grouping**: Optimizers can split gate parameters (`lr=1e-4`) and specialist parameters (`lr=2e-5`).

#### Feature 7: Multi-Objective Loss Formulation
- **T1.7.1 — Component Aggregation**: $\mathcal{L}_{\text{tot}} = \lambda_1 \mathcal{L}_1 + \lambda_2 (1-\text{SSIM}) + \lambda_3 \mathcal{L}_{\text{CE}} + \lambda_4 \mathcal{L}_{\text{balance}}$ matches manual summation.
- **T1.7.2 — Perfect Reconstruction Bound**: When $\hat{x} = x$, $\mathcal{L}_1 = 0$ and $(1-\text{SSIM}) = 0$.
- **T1.7.3 — SSIM Non-negativity**: $(1 - \text{SSIM})$ remains strictly $\ge 0$ for all valid images in $[0, 1]$.
- **T1.7.4 — Auxiliary Cross-Entropy Loss**: Correctly maps corruption labels $y \in \{0, 1, 2, 3\}$ to gating logits $z$.
- **T1.7.5 — Weight Coefficient Sensitivity**: Varying $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ scales individual loss gradients proportionally.

#### Feature 8: Balance Regularizers Formulation
- **T1.8.1 — L2 Deviation Minimum**: $\mathcal{L}_{\text{L2}} = \sum_{k=1}^4 (\bar{w}_k - 0.25)^2$ attains minimum $0.0$ when $\bar{w} = [0.25, 0.25, 0.25, 0.25]$.
- **T1.8.2 — L2 Deviation Penalty on Dominance**: Completely collapsed routing $\bar{w} = [1, 0, 0, 0]$ yields penalty $(0.75^2 + 3 \times 0.25^2) = 0.5625 + 0.1875 = 0.75$.
- **T1.8.3 — Entropy Maximization**: $\mathcal{L}_{\text{entropy}} = \sum_{k=1}^4 \bar{w}_k \log(\bar{w}_k + \epsilon)$ reaches minimum $-\log(4) \approx -1.3863$ at uniformity.
- **T1.8.4 — Switch Load Balance**: Cross-term $4 \sum f_k \bar{w}_k$ penalizes alignment of argmax assignments with large soft probabilities.
- **T1.8.5 — Regularizer Gradient Flow**: Balance loss backpropagates valid non-zero gradients to gating logits $z$.

#### Feature 9: Optuna Pruning & Collapse Detection
- **T1.9.1 — Dominance Prune Trigger**: Synthetic distribution with $\bar{w}_{\text{salt}} = 0.92 > 0.90$ triggers `TrialPruned`.
- **T1.9.2 — Starvation Prune Trigger**: Synthetic distribution with $\bar{w}_{\text{blur}} = 0.015 < 0.02$ triggers `TrialPruned`.
- **T1.9.3 — Healthy Distribution Survival**: Balanced distribution $\bar{w} = [0.30, 0.25, 0.20, 0.25]$ passes without pruning.
- **T1.9.4 — Boundary Invariance**: $\bar{w} = [0.90, 0.033, 0.033, 0.034]$ remains unpruned at threshold boundary.
- **T1.9.5 — Multi-Epoch Callback Integrity**: Callback accurately aggregates routing statistics across sequential evaluation steps.

#### Feature 10: Routing Analysis & Confusion Matrix
- **T1.10.1 — 4x4 Confusion Matrix Geometry**: Routing matrix has shape `(4, 4)` where rows index ground truth corruptions and columns index expert weights.
- **T1.10.2 — Row Sum Conservation**: Every row in the routing confusion matrix sums to $1.0 \pm 10^{-5}$.
- **T1.10.3 — Diagonal Dominance Verification**: Healthy routing matrix demonstrates maximum weight along diagonal ($w_{k, k} > w_{k, j}$ for $j \ne k$).
- **T1.10.4 — Monotonic Severity Progression**: Increasing noise density $p$ monotonically increases specialist routing allocation.
- **T1.10.5 — Expert Health Audit**: Verifies every expert achieves $\ge 5\%$ average allocation across validation dataset.

#### Feature 11: ONNX Export & Numerical Parity
- **T1.11.1 — Structural Model Checker**: Exported ONNX graph passes `onnx.checker.check_model` without schema errors (opset 17).
- **T1.11.2 — Input/Output Tensor Naming**: Graph exposes `input_image` [B, 3, 128, 128] and outputs `restored_image` [B, 3, 128, 128], `routing_weights` [B, 4].
- **T1.11.3 — Dynamic Batch Axis**: ORT session accepts arbitrary batch sizes $B \in \{1, 3, 6\}$ without reshape failure.
- **T1.11.4 — Reconstruction Numerical Parity**: PyTorch vs ONNX Runtime absolute error $\max |\hat{x}_{\text{pt}} - \hat{x}_{\text{ort}}| < 1 \times 10^{-5}$.
- **T1.11.5 — Routing Weights Numerical Parity**: PyTorch vs ONNX Runtime absolute error $\max |w_{\text{pt}} - w_{\text{ort}}| < 1 \times 10^{-5}$.

---

### Tier 2: Boundary & Corner Cases

- **T2.1 — Extreme Low Temperature ($\tau = 1 \times 10^{-5}$)**: Clamping ensures $\tau \ge 0.05$, preventing NaN/Inf in logits exponential.
- **T2.2 — Extreme High Temperature ($\tau = 100.0$)**: Softmax does not underflow; routing smoothly approaches $[0.25, 0.25, 0.25, 0.25]$.
- **T2.3 — Singleton Batch Processing ($B = 1$)**: Dynamic dimension expansion $w[:, k, \text{None}, \text{None}, \text{None}]$ handles singleton batch without dimension collapse.
- **T2.4 — Prime Number Batch Size ($B = 7$)**: Ensures parallel branches and batchnorm do not assume powers-of-two.
- **T2.5 — All-Zero Input Tensor (Pure Black Image)**: Soft MoE processes black image without zero-division in normalization or SSIM loss.
- **T2.6 — All-One Input Tensor (Pure White Image)**: Soft MoE processes white image without saturation overflow.
- **T2.7 — Saturated Noise Extreme ($p = 1.0$)**: Severe salt-and-pepper corruption (100% pixel corruption) processes without crashing.
- **T2.8 — Severe Full-Image Occlusion ($100\%$ box area)**: Fully occluded image passes through MoE cleanly.
- **T2.9 — Floating-Point Dynamic Range Outliers**: Inputs slightly outside $[0.0, 1.0]$ (e.g. $[-0.05, 1.05]$) do not cause unhandled exceptions.
- **T2.10 — Gradient Explosion Safeguard**: Simulated large loss yields finite gradients when clipped with `clip_grad_norm_ <= 1.0`.

---

### Tier 3: Cross-Feature Interactions

- **T3.1 — Warm-up to Joint Fine-Tuning Transition**: Verifies that transition preserves learned gate parameters while enabling specialist autograd.
- **T3.2 — Differentiable Convex Blending Multi-Branch Flow**: Backward pass on reconstruction loss computes non-zero gradients simultaneously in Gate, Salt specialist, Blur specialist, and Occlusion specialist.
- **T3.3 — Cross-Entropy vs Reconstruction Loss Interaction**: Auxiliary CE loss reinforces correct gate routing without destabilizing image reconstruction gradients.
- **T3.4 — Balance Regularizer Countering Dominance**: When artificial corruptions bias the gate toward one specialist, balance regularizer produces opposing gradient on logits.
- **T3.5 — ExportWrapper Encapsulation**: Wrapper fixes temperature parameter $\tau=1.0$, exposing single-input signature compatible with ONNX tracer.
- **T3.6 — ONNX Execution Equivalence on Blended Output**: Verifies that dynamic 5D tensor multiplication in ONNX produces identical results to PyTorch eager execution.

---

### Tier 4: Real-World Scenarios

- **T4.1 — Clean Oxford-IIIT Pet Restoration**: Clean test image processed by Soft MoE achieves PSNR $> 30.0\text{ dB}$, confirming clean preservation.
- **T4.2 — Salt-and-Pepper Pet Restoration**: Pet image corrupted with $p=0.08$ noise is primarily routed to $w_{\text{salt}}$, achieving PSNR $> 18.0\text{ dB}$.
- **T4.3 — Gaussian Blur Pet Restoration**: Pet image degraded with $(k=5, \sigma=1.5)$ is routed to $w_{\text{blur}}$, improving visual clarity.
- **T4.4 — Rectangular Occlusion Pet Restoration**: Pet image with 20% occlusion is routed to $w_{\text{occ}}$, reconstructing masked areas.
- **T4.5 — Baseline Superiority Check**: Evaluates that Soft MoE restoration outperforms the unrouted identity baseline across all corrupted samples.

---

## 3. Expected Output Derivation & Authoritative Oracles

| Test Case Category | Expected Output Origin | Authoritative Oracle |
| :--- | :--- | :--- |
| Routing Weight Probability Constraint | Mathematical Axiom of Softmax | $\sum_{k=1}^4 w_k = 1.0$, $w_k \in [0, 1]$ |
| Temperature Clamping | Contract Specification (`PROJECT.md` § Interface Contracts) | $\tau_{\text{eff}} = \max(\tau, 0.05)$ |
| Identity Bypass Fidelity | Mathematical Identity Function $f(x) = x$ | $\hat{x} \equiv \tilde{x}$, $\text{MSE} = 0.0$ |
| Checkpoint Parameter Integrity | Pre-trained Checkpoints in `checkpoints/task2/` | Exact parameter shapes and key mappings |
| Balance Regularizer Optimal Value | Quadratic Formulation $\sum (\bar{w}_k - 0.25)^2$ | Minimized at $0.0$ when $\bar{w}_k = 0.25$ |
| Pruning Thresholds | Optuna Specification in `docs/plans/task3-soft-moe.md` § Step 6 | Prune if $\max \bar{w} > 0.90$ or $\min \bar{w} < 0.02$ |
| ONNX Numerical Parity | PyTorch FP32 Eager Execution vs ONNX Runtime CPU | Absolute difference $< 1 \times 10^{-5}$ |
| Dataset Degradations | Core Degradation Engine `src.shared.corruptions` | Standardized parameter regimes |

---

## 4. Test Execution & Environment

### Environment Requirements
- **Python**: 3.11+
- **PyTorch**: 2.6.0+cu124
- **ONNX**: 1.23.0
- **ONNX Runtime**: 1.30.0
- **Pytest**: 9.1.1
- **Hardware Acceleration**: CPU & CUDA (NVIDIA GeForce RTX 3050 6GB)

### Execution Command
```powershell
uv run pytest -v tests/test_task3_e2e.py
```

### Selective Execution by Tier
```powershell
# Run Tier 1 Feature Coverage
uv run pytest -v -k "test_tier1" tests/test_task3_e2e.py

# Run Tier 2 Boundaries & Corners
uv run pytest -v -k "test_tier2" tests/test_task3_e2e.py

# Run Tier 3 Interactions
uv run pytest -v -k "test_tier3" tests/test_task3_e2e.py

# Run Tier 4 Real-World Pipeline
uv run pytest -v -k "test_tier4" tests/test_task3_e2e.py
```

---

## 5. Failure Taxonomy & Bug Escalation Protocol

| Error Category | Symptoms | Responsible Subsystem | Escalation Route |
| :--- | :--- | :--- | :--- |
| **Probability Violation** | $\sum w_k \ne 1.0$ or $w_k < 0$ | GatingNetwork Softmax Layer | Worker M1 / `src/task3/gate.py` |
| **Temperature Divergence** | NaN/Inf when $\tau < 0.05$ | Clamping Safeguard in Gate | Worker M1 / `src/task3/gate.py` |
| **Gradient Detachment** | `param.grad is None` after backward | Blending or Specialist forward pass | Worker M1 / `src/task3/moe_model.py` |
| **Identity Degradation** | $\hat{x} \ne \tilde{x}$ when $w_1 = 1.0$ | Convex Blending / Identity bypass | Worker M1 / `src/task3/moe_model.py` |
| **Checkpoint Shape Mismatch** | `RuntimeError: size mismatch` | Component Checkpoint Loader | Worker M1 / `src/task3/moe_model.py` |
| **Loss Negative Bound** | $(1 - \text{SSIM}) < 0$ or NaN loss | Multi-Objective Loss Formulation | Worker M2 / `src/task3/train.py` |
| **Pruning Misfire** | Healthy trials pruned prematurely | Optuna Pruning Callback | Worker M3 / `src/task3/optuna_search.py` |
| **ONNX Parity Exceeded** | $\max |\hat{x}_{\text{pt}} - \hat{x}_{\text{ort}}| \ge 10^{-5}$ | ONNX Exporter / Opset config | Worker M5 / `src/task3/export_onnx.py` |
