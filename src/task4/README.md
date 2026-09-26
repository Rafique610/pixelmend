# Task 4: Style-Conditioned Face-to-Sketch Synthesis (Conditional GAN)

## Overview
Paired facial photograph-to-sketch generation using a conditional Generative Adversarial Network conditioned on three categorical sketch styles from the FS2K dataset.

## Architecture
- **Generator**: U-Net encoder-decoder with learned style embedding conditioning: $\hat{y} = G(x, s)$.
- **Discriminator**: PatchGAN architecture operating on photograph, style, and sketch triplets: $D(x, y, s)$.
- **Objective**:
  $$\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \lambda_{L1} \mathcal{L}_{L1}(y, G(x, s))$$
  $$\mathcal{L}_D = \text{BCEWithLogits}(D_{\text{real}}, 1) + \text{BCEWithLogits}(D_{\text{fake}}, 0)$$

## Modules
- `generator.py`: Style-conditioned U-Net generator.
- `discriminator.py`: Style-conditioned PatchGAN discriminator.
- `augmentation.py`: Synchronized paired spatial transformations.
- `train.py`: Alternating adversarial training with visual progress tracking.
- `optuna_search.py`: Optuna study for GAN hyperparameters (`task4-cgan`).
- `evaluate.py`: Test set evaluation (L1, SSIM, LPIPS, FID) across styles.
- `export_onnx.py`: ONNX export of generator model for inference.
