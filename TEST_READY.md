# TEST_READY — Task 3 Soft Mixture-of-Experts (MoE) Image Restoration

**Test Suite Status:** READY & FULLY VERIFIED (100% PASS RATE)  
**Date:** 2026-10-04  
**Primary Test Target:** `tests/test_task3_e2e.py`  
**Execution Command:** `uv run pytest -v tests/test_task3_e2e.py`  
**All Task 3 Tests:** `uv run pytest -v tests/test_task3_moe.py tests/test_task3_e2e.py`  

---

## 1. Test Suite Summary

The end-to-end (E2E) test suite for Task 3 Soft Mixture-of-Experts (MoE) Image Restoration has been designed, implemented, and executed according to the 4-Tier Testing Methodology specified in `TEST_INFRA.md`.

| Metric | E2E Suite (`tests/test_task3_e2e.py`) | Combined Task 3 Suite (`moe` + `e2e`) |
| :--- | :--- | :--- |
| **Total Test Cases** | **76** | **94** |
| **Passed** | **76 (100%)** | **94 (100%)** |
| **Failed** | **0** | **0** |
| **Skipped** | **0** | **0** |
| **Execution Duration** | **19.40s** | **25.63s** |
| **Ruff Lint Violations** | **0** (Clean) | **0** (Clean) |

---

## 2. Four-Tier Coverage Matrix

### Tier 1: Feature Coverage (55 Tests across M1–M5 Features)
- **Feature 1: Gating Network Architecture (5 tests)**
  - `test_tier1_feat1_gate_output_topology`: Output shapes $(B, 4)$ for routing weights and logits.
  - `test_tier1_feat1_gate_softmax_normalization`: Simplex property $\sum_{k=1}^4 w_k = 1.0 \pm 10^{-6}$.
  - `test_tier1_feat1_gate_probability_bounds`: All $w_{i, k} \in [0.0, 1.0]$.
  - `test_tier1_feat1_gate_checkpoint_loading_real`: Real Task 2 classifier checkpoint deserialization.
  - `test_tier1_feat1_gate_backbone_feature_extraction`: Spatial pooling to $(B, 256)$ features.
- **Feature 2: Temperature Scaling & Numerical Clamping (5 tests)**
  - `test_tier1_feat2_temperature_standard_tau_1`: $\tau=1.0$ matches standard softmax.
  - `test_tier1_feat2_temperature_smoothing_tau_high`: $\tau=5.0$ increases routing entropy.
  - `test_tier1_feat2_temperature_sharpening_tau_low`: $\tau=0.1$ sharpens allocation toward argmax.
  - `test_tier1_feat2_temperature_clamping_safeguard`: $\tau < 0.05$ clamped to $0.05$, zero NaNs.
  - `test_tier1_feat2_temperature_equal_logits_invariance`: Invariance on equal logits.
- **Feature 3: SoftMoE Container Module (5 tests)**
  - `test_tier1_feat3_moe_submodules_composition`: Composition with gate and 3 specialists.
  - `test_tier1_feat3_moe_forward_three_tuple_return`: Returns `(restored, routing_weights, logits)`.
  - `test_tier1_feat3_moe_spatial_dimension_fidelity`: Exact spatial preservation $(B, 3, 128, 128)$.
  - `test_tier1_feat3_moe_multi_batch_evaluation`: Multi-batch evaluation $B \in \{1, 2, 4\}$.
  - `test_tier1_feat3_moe_eval_vs_train_mode`: Mode switching between `train()` and `eval()`.
- **Feature 4: Differentiable Convex Blending (5 tests)**
  - `test_tier1_feat4_convex_combination_identity`: Mathematical convex combination $\sum w_k b_k$.
  - `test_tier1_feat4_end_to_end_gradient_propagation`: Non-zero gradients to gate and all active specialists.
  - `test_tier1_feat4_gate_logit_perturbation_sensitivity`: Perturbations in logits change output tensor.
  - `test_tier1_feat4_one_hot_orthogonal_isolation`: One-hot weights isolate specific branch outputs.
  - `test_tier1_feat4_blending_preserves_finite_range`: Output bounds within $[0.0, 1.0]$.
- **Feature 5: Identity Pass-Through Branch (5 tests)**
  - `test_tier1_feat5_identity_bit_exact_bypass`: $w = [1, 0, 0, 0] \implies \hat{x} \equiv \tilde{x}$ with $0.0$ MSE.
  - `test_tier1_feat5_identity_zero_parameters`: Identity branch has zero trainable parameters.
  - `test_tier1_feat5_identity_zero_compute_overhead`: Direct memory assignment without extra FLOPs.
  - `test_tier1_feat5_identity_unmodified_dynamic_range`: Dynamic range $[\min, \max]$ preserved bit-exact.
  - `test_tier1_feat5_identity_high_frequency_fidelity`: High-frequency textures preserved intact.
- **Feature 6: Pre-trained Checkpoint Integration & Freezing (5 tests)**
  - `test_tier1_feat6_pretrained_load_all_four_components`: Loads all 4 Task 2 checkpoints into SoftMoE.
  - `test_tier1_feat6_freeze_specialists_toggle`: `set_experts_frozen(True)` freezes all specialists.
  - `test_tier1_feat6_unfreeze_specialists_toggle`: `set_experts_frozen(False)` restores trainable state.
  - `test_tier1_feat6_warmup_specialist_weights_isolation`: Optimizer step on gate leaves specialists untouched.
  - `test_tier1_feat6_differential_lr_parameter_groups`: Optimizer configured with differential LRs (1e-4 / 2e-5).
- **Feature 7: Multi-Objective Loss Formulation (5 tests)**
  - `test_tier1_feat7_loss_formula_summation`: Total loss summation $\lambda_1 L_1 + \lambda_2 (1-\text{SSIM}) + \lambda_3 L_{\text{CE}} + \lambda_4 L_{\text{bal}}$.
  - `test_tier1_feat7_loss_zero_on_identical_reconstruction`: $L_1 = 0$ and $(1-\text{SSIM}) = 0$ when $\hat{x} = x$.
  - `test_tier1_feat7_loss_ssim_non_negativity`: Structural dissimilarity $\ge 0$ for all valid images.
  - `test_tier1_feat7_loss_auxiliary_ce_alignment`: CE loss reinforces corruption label alignment.
  - `test_tier1_feat7_loss_gradient_scaling_proportionality`: Scaling $\lambda_1$ scales gradients proportionally.
- **Feature 8: Balance Regularizers Formulations (5 tests)**
  - `test_tier1_feat8_l2_regularizer_zero_at_uniform`: L2 deviation reaches $0.0$ at uniform $[0.25, 0.25, 0.25, 0.25]$.
  - `test_tier1_feat8_l2_regularizer_maximum_at_complete_collapse`: L2 penalty reaches $0.75$ at complete collapse $[1, 0, 0, 0]$.
  - `test_tier1_feat8_entropy_regularizer_minimum_at_uniform`: Negative entropy reaches $-\log(4)$ at uniform.
  - `test_tier1_feat8_switch_load_balance_formulation`: Evaluates $4 \sum f_k \bar{w}_k$ load balance.
  - `test_tier1_feat8_balance_regularizer_backpropagation`: Balance loss backpropagates to routing logits.
- **Feature 9: Optuna Pruning & Collapse Detection (5 tests)**
  - `test_tier1_feat9_optuna_pruner_dominance_trigger`: $\max \bar{w}_k > 0.90$ triggers collapse pruning.
  - `test_tier1_feat9_optuna_pruner_starvation_trigger`: $\min \bar{w}_k < 0.02$ triggers collapse pruning.
  - `test_tier1_feat9_optuna_pruner_healthy_passes`: Balanced distributions pass without pruning.
  - `test_tier1_feat9_optuna_pruner_exact_threshold_boundary`: Handles exact boundary values with float epsilon.
  - `test_tier1_feat9_optuna_pruner_multi_batch_aggregation`: Running mean maintains prune decisions.
- **Feature 10: Routing Analysis & Confusion Matrix (5 tests)**
  - `test_tier1_feat10_confusion_matrix_shape_4x4`: 4x4 matrix dimensions for corruption classes and experts.
  - `test_tier1_feat10_confusion_matrix_row_stochastic`: Row sums equal $1.0 \pm 10^{-5}$.
  - `test_tier1_feat10_diagonal_dominance_check`: Diagonal dominance verifies corruption specificity.
  - `test_tier1_feat10_monotonic_severity_progression`: Severity increase yields monotonic routing weight increase.
  - `test_tier1_feat10_expert_health_audit_threshold`: Audits all experts $\ge 5\%$ minimum allocation.
- **Feature 11: Atomic ONNX Export & Equivalence (5 tests)**
  - `test_tier1_feat11_onnx_export_wrapper_signature`: ExportWrapper fixes $\tau$ and exposes clean signature.
  - `test_tier1_feat11_onnx_export_graph_validation`: `onnx.checker.check_model` passes under opset 17.
  - `test_tier1_feat11_onnx_dynamic_batching_support`: Accepts arbitrary batch sizes $B \in \{1, 3, 5\}$.
  - `test_tier1_feat11_onnx_restoration_numerical_parity`: $\max |\hat{x}_{\text{pt}} - \hat{x}_{\text{ort}}| < 1 \times 10^{-5}$.
  - `test_tier1_feat11_onnx_routing_weights_numerical_parity`: $\max |w_{\text{pt}} - w_{\text{ort}}| < 1 \times 10^{-5}$.

---

### Tier 2: Boundary & Corner Cases (10 Tests)
- `test_tier2_extreme_low_temperature_clamping`: Clamping $\tau=10^{-6} \to 0.05$ prevents NaN/Inf.
- `test_tier2_extreme_high_temperature_smoothing`: $\tau=100.0$ approaches uniform $[0.25, 0.25, 0.25, 0.25]$.
- `test_tier2_negative_temperature_handling`: $\tau=-0.5$ clamped to $0.05$.
- `test_tier2_singleton_batch_processing`: Singleton batch $B=1$ restores without shape collapse.
- `test_tier2_prime_batch_size`: Prime batch $B=7$ evaluates correctly.
- `test_tier2_all_zero_tensor_pure_black`: Pure black input restores without division-by-zero.
- `test_tier2_all_one_tensor_pure_white`: Pure white input restores without numerical saturation.
- `test_tier2_extreme_noise_density`: $p=1.0$ (100% noise) processes cleanly.
- `test_tier2_full_image_occlusion`: 100% full image occlusion processes without crash.
- `test_tier2_gradient_clipping_guard`: Gradient clipping with `max_norm=1.0` bounds gradients under extreme loss.

---

### Tier 3: Cross-Feature Interactions (6 Tests)
- `test_tier3_warmup_to_joint_transition_preservation`: Specialists frozen in warm-up retain exact weights; unfreezing enables joint gradient flow.
- `test_tier3_simultaneous_multi_branch_gradient_flow`: Blended reconstruction loss delivers gradients to Gate and all 3 specialists simultaneously.
- `test_tier3_gate_logit_sensitivity_direct_blending`: Manually shifting gate logits directly and smoothly alters composite restoration output.
- `test_tier3_balance_loss_counters_collapse_pressure`: Balance regularizer creates counter-gradient opposing expert domination.
- `test_tier3_onnx_export_wrapper_fixed_tau_eager_match`: ExportWrapper eager mode matches standalone SoftMoE.
- `test_tier3_onnx_blended_restoration_parity_multi_batch`: PyTorch vs ONNX Runtime parity verified across dynamic batch sizes $B \in \{1, 2, 4\}$.

---

### Tier 4: Real-World Scenarios (5 Tests)
- `test_tier4_clean_pet_restoration_psnr`: Clean Oxford-IIIT Pet image through loaded SoftMoE retains PSNR $\ge 20.0$ dB.
- `test_tier4_salt_and_pepper_pet_restoration`: S&P corrupted pet image ($p=0.08$) routes primarily to Salt specialist ($w > 0.50$) and restores.
- `test_tier4_gaussian_blur_pet_restoration`: Gaussian blurred pet image ($k=5, \sigma=1.5$) routes to Blur specialist ($w > 0.50$) and restores.
- `test_tier4_occlusion_pet_restoration`: 20% occluded pet image routes to Occlusion specialist ($w > 0.50$) and restores.
- `test_tier4_restoration_superiority_over_unrouted_identity`: SoftMoE restoration achieves strictly higher PSNR than unrouted corrupted input on real pet image.

---

## 3. How to Execute Tests

```powershell
# Run the complete Task 3 E2E test suite (76 tests)
uv run pytest -v tests/test_task3_e2e.py

# Run all Task 3 test suites (94 tests)
uv run pytest -v tests/test_task3_moe.py tests/test_task3_e2e.py

# Run only a specific tier
uv run pytest -v -k "test_tier1" tests/test_task3_e2e.py
uv run pytest -v -k "test_tier2" tests/test_task3_e2e.py
uv run pytest -v -k "test_tier3" tests/test_task3_e2e.py
uv run pytest -v -k "test_tier4" tests/test_task3_e2e.py
```

---

## 4. Conclusion & Next Steps

All 76 end-to-end opaque-box test cases for Task 3 are implemented, conform to all repository invariants (`core.md`, `ml.md`, `pyproject.toml`), pass 100% of unit assertions in <20 seconds, and pass strict Ruff linting with zero violations. Task 3 Soft MoE test infrastructure is fully verified and ready for progressive milestone integration.
