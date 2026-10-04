# Report Artifact Map

Where each report section's evidence lives. All paths relative to repo root. JSON/CSV/MD are committed; PNG figures are git-ignored (regenerate with `task eval-taskN`).

| Report section | Numbers (JSON / CSV / MD) | Figures (PNG) |
| :--- | :--- | :--- |
| Data & corruption pipeline | `manifests/` | `results/oxford_pets_sample_grid.png`, `results/fs2k_sample_grid.png`, `results/corruption_verification_grid.png` |
| Task 1 — architecture, loss (alpha), Optuna | `results/task1/metrics/best_hyperparams.json`, `baseline_history.json`, `final_history.json` | `results/task1/visualizations/alpha_spectrum_analysis.png`, `optuna_optimization_history.png`, `optuna_param_importances.png`, `baseline_training_curves.png`, `final_training_curves.png` |
| Task 1 — test results & failures | `results/task1/metrics_summary.json`, `metrics/test_metrics.json`, `metrics/test_summary_table.md` | `results/task1/visualizations/qualitative_comparison_grid.png`, `failure_cases_analysis.png` |
| Task 2 — classifier research & tuning | `results/task2/classifier_architecture_benchmark.json`, `classifier_best_hyperparams.json`, `classifier-baseline_metrics.json` | `classifier-baseline_confusion_matrix.png`, `classifier-baseline_curves.png`, `classifier_optuna_*.png` |
| Task 2 — specialists | `results/task2/specialist_architecture_benchmark.json`, `specialists_best_hyperparams.json`, `specialists_baseline_metrics.json` | `specialists_baseline_curves.png`, `specialists_optuna_*.png`, `specialists_sample_reconstructions.png` |
| Task 2 — oracle vs predicted routing | `results/task2/router_benchmark.json`, `metrics_summary.{json,csv}`, `test_summary_table.md` | `results/task2/visuals/qualitative_comparison_grid.png`, `routing_failure_cases.png` |
| Task 3 — Soft MoE training, Optuna, regularisers | `results/task3/baseline_train_metrics.json`, `final_train_metrics.json`, `final_val_metrics.json`, `balance_regularizers_benchmark.json`, `challenger2_m2_evaluation.json` | `results/task3/figures/optuna_moe_*.png` |
| Task 3 — routing analysis | `results/task3/routing_analysis_summary.json`, `routing_confusion_matrix.csv` | `figures/routing_heatmap.png`, `routing_galleries.png`, `severity_routing_trends.png` |
| Task 1 vs 2 vs 3 comparison | `results/task3/test_benchmark_comparison.csv`, `test_evaluation_summary.json` | `figures/qualitative_comparison_12.png`, `failure_cases_4.png` |
| Task 4 — architecture & conditioning research | `results/task4/architecture_conditioning_benchmark.json`, `discriminator_benchmark.json`, `optuna_best_params.json` | — |
| Task 4 — training | `results/task4/baseline_train_metrics.json`, `baseline_val_metrics.json`, `final_train_metrics.json` | `results/task4/visualizations/*progression*.png` |
| Task 4 — test results | `results/task4/test_metrics.json` | `sample_results_grid.png`, `style_comparison_grid.png`, `failure_cases_analysis.png` |
| ONNX export & parity | `results/task{1,2,3,4}/**/onnx_parity_benchmark.json` | — |
| Deployment CPU benchmark | `results/app/cpu_benchmark.json` (`scripts/benchmark_cpu.py`) | — |
| App design | `results/app/stitch_design_spec.json` | `results/app/stitch_*.jpg` |
| Application verification | `scripts/smoke_test.py` (25 live checks), `tests/test_backend_*.py` (63 tests) | — |

## Honest-reporting notes

- Task 3 vs Task 1/2 comparison uses a 20 % subsample (7,340 images); Task 1 and Task 2 also have full 36,690-image numbers.
- Task 2's high mean PSNR comes from identity bypass on clean inputs (PSNR capped at 80 dB). Compare per-corruption rows, where Task 1 and Task 3 are higher on corrupted classes.
- Task 4 PSNR/SSIM are low in absolute terms because sketch styles are stylistic; LPIPS and the visual grids are the more informative evidence.
- Latencies are from one CPU machine; see `cpu_benchmark.json` for the exact platform.
