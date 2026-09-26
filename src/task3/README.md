# Task 3: Jointly Trained Soft Mixture-of-Experts (MoE)

## Overview
A differentiable soft mixture-of-experts model transforming the discrete routing of Task 2 into a continuous convex combination of expert outputs:
$$\hat{x} = w_1 \tilde{x} + w_2 A_{\text{salt}}(\tilde{x}) + w_3 A_{\text{blur}}(\tilde{x}) + w_4 A_{\text{occlusion}}(\tilde{x})$$
where $w = \text{softmax}(G(\tilde{x})/\tau)$.

## Training Procedure
- **Stage 1 (Warm-up)**: Initialize gate with Task 2 classifier and experts with Task 2 specialists. Freeze experts, train gate.
- **Stage 2 (Joint Fine-Tuning)**: Unfreeze experts and train end-to-end with joint objective:
  $$\mathcal{L}_{\text{joint}} = \lambda_1 \mathcal{L}_1 + \lambda_2 (1 - \text{SSIM}) + \lambda_3 \mathcal{L}_{\text{CE}} + \lambda_4 \mathcal{L}_{\text{balance}}$$

## Modules
- `gate.py`: Temperature-controlled softmax gating network.
- `moe_model.py`: Differentiable SoftMoE container combining identity branch and 3 specialists.
- `train.py`: Two-phase warm-up and joint fine-tuning orchestrator.
- `optuna_search.py`: Optuna study for fine-tune LR, temperature $\tau$, and loss weights (`task3-moe-joint`).
- `routing_analysis.py`: Gating weight heatmaps, expert distribution, and collapse checks.
- `evaluate.py`: Test benchmark against Task 1 and Task 2.
- `export_onnx.py`: End-to-end unified ONNX graph export.
