### Test Set Evaluation: Per-Corruption Summary Table

| Corruption Type | Mean PSNR (dB) | Mean SSIM | Mean MAE (L1) | Mean MSE (L2) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean** | 20.90 dB | 0.6178 | 0.0690 | 0.00998 |
| **Salt And Pepper** | 20.75 dB | 0.6067 | 0.0703 | 0.01022 |
| **Gaussian Blur** | 20.92 dB | 0.6117 | 0.0693 | 0.00987 |
| **Occlusion** | 18.56 dB | 0.5541 | 0.0847 | 0.01648 |
| **Overall Mean** | **20.16 dB** | **0.5935** | **0.0742** | **0.01197** |

### Test Set Evaluation: Per-Severity Breakdown Table

| Condition | Severity Level | Target Parameters | PSNR (dB) | SSIM | MAE (L1) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `clean` | Severity 0 | Clean / Uncorrupted | 20.90 dB | 0.6178 | 0.0690 |
| `sp_mild` | Severity 1 | p = 0.03 | 20.87 dB | 0.6149 | 0.0693 |
| `sp_medium` | Severity 2 | p = 0.08 | 20.79 dB | 0.6083 | 0.0700 |
| `sp_severe` | Severity 3 | p = 0.15 | 20.59 dB | 0.5968 | 0.0716 |
| `blur_mild` | Severity 1 | k = 3, sigma = 0.7 | 20.94 dB | 0.6174 | 0.0689 |
| `blur_medium` | Severity 2 | k = 5, sigma = 1.5 | 20.95 dB | 0.6139 | 0.0690 |
| `blur_severe` | Severity 3 | k = 7, sigma = 2.5 | 20.86 dB | 0.6040 | 0.0699 |
| `occl_mild` | Severity 1 | 1 box (~10%) | 19.69 dB | 0.5870 | 0.0758 |
| `occl_medium` | Severity 2 | 2 boxes (~20%) | 18.72 dB | 0.5587 | 0.0827 |
| `occl_severe` | Severity 3 | 3 boxes (~35%) | 17.26 dB | 0.5166 | 0.0955 |