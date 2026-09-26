# Task 2: Corruption Classification & Hard-Routed Specialist Autoencoders

## Overview
A hard-routing restoration pipeline consisting of:
1. **4-class Corruption Classifier**: Predicts whether input is clean, salt-and-pepper, Gaussian blur, or rectangular occlusion.
2. **Identity Bypass**: Leaves clean inputs unprocessed.
3. **3 Specialist Autoencoders**: Independently trained on salt-and-pepper, blur, and occlusion respectively.

## Architecture
- $p = C(\tilde{x}) \in \Delta^3, \quad r = \arg\max_k p_k$
- If $r = \text{clean} \implies \hat{x} = \tilde{x}$
- If $r = \text{corruption}_k \implies \hat{x} = A_k(\tilde{x})$
- Evaluated in Oracle Routing (ground-truth label) vs Predicted Routing (classifier label).

## Modules
- `classifier.py`: Convolutional 4-class classifier.
- `specialist.py`: Specialist autoencoder architecture.
- `train_classifier.py`: Balanced-batch classifier training.
- `train_specialists.py`: Independent specialist training runs.
- `router.py`: Hard routing dispatch and inference pipeline.
- `optuna_classifier.py`: Optuna study for classifier (`task2-classifier`).
- `optuna_specialists.py`: Optuna study for specialists (`task2-specialists`).
- `evaluate.py`: Oracle vs predicted routing benchmarks & failure analysis.
- `export_onnx.py`: ONNX export for classifier and 3 specialists (4 models).
