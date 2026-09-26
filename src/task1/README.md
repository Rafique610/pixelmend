# Task 1: Universal Multi-Corruption Denoising Autoencoder

## Overview
A single convolutional autoencoder capable of restoring clean images and images affected by three corruption types without prior knowledge of the corruption:
1. Salt-and-pepper noise
2. Gaussian blur
3. Rectangular occlusion

## Architecture
- **Encoder**: Progressive spatial downsampling ($128 \times 128 \to \text{bottleneck}$) with increasing channel depth.
- **Bottleneck**: Meaningful compressed latent space.
- **Decoder**: Progressive spatial upsampling reconstructing RGB $128 \times 128$.
- **Loss**: Combined pixel reconstruction and structural similarity:
  $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_1(x, \hat{x}) + (1 - \alpha)(1 - \text{SSIM}(x, \hat{x}))$$

## Modules
- `encoder.py`: Convolutional encoder backbone.
- `decoder.py`: Convolutional decoder backbone.
- `autoencoder.py`: Combined PyTorch module.
- `train.py`: Training loop with dynamic corruptions and validation against deterministic manifest.
- `optuna_search.py`: Optuna hyperparameter optimization study (`task1-universal-ae`).
- `evaluate.py`: Multi-severity test evaluation with absolute error map generation.
- `export_onnx.py`: ONNX export and parity verification.
