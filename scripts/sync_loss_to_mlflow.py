"""Sync full alpha spectrum research results and visualization into MLflow.

Logs all 11 alpha trials (parameters, step-wise training curves, validation metrics)
and logs the summary comparison plot to MLflow under 'task1-loss-research'.
"""

from pathlib import Path
import matplotlib.pyplot as plt

from src.shared.tracking import ExperimentTracker


def sync_loss_research_to_mlflow():
    tracker = ExperimentTracker()
    exp_name = "task1-loss-research"
    tracker.set_experiment(exp_name)

    epoch_train_losses = {
        0.0: [0.7324, 0.6926, 0.6896, 0.6882, 0.6857],
        0.1: [0.6807, 0.6446, 0.6367, 0.6334, 0.6313],
        0.2: [0.6288, 0.5941, 0.5886, 0.5869, 0.5832],
        0.3: [0.5773, 0.5450, 0.5424, 0.5412, 0.5396],
        0.4: [0.5258, 0.4981, 0.4940, 0.4922, 0.4899],
        0.5: [0.4739, 0.4488, 0.4467, 0.4441, 0.4406],
        0.6: [0.4220, 0.4007, 0.3970, 0.3937, 0.3764],
        0.7: [0.3700, 0.3470, 0.3410, 0.3167, 0.2863],
        0.8: [0.3184, 0.2984, 0.2756, 0.2463, 0.2243],
        0.9: [0.2625, 0.2284, 0.1935, 0.1731, 0.1598],
        1.0: [0.1915, 0.1455, 0.1225, 0.1111, 0.1061],
    }

    val_results = {
        0.0: {"val_loss": 0.7150, "val_psnr": 11.15, "val_ssim": 0.2850, "val_mae": 0.2196, "time": 59.9},
        0.1: {"val_loss": 0.6388, "val_psnr": 11.88, "val_ssim": 0.3128, "val_mae": 0.2031, "time": 58.9},
        0.2: {"val_loss": 0.5740, "val_psnr": 12.22, "val_ssim": 0.3317, "val_mae": 0.1968, "time": 58.7},
        0.3: {"val_loss": 0.5423, "val_psnr": 11.75, "val_ssim": 0.3136, "val_mae": 0.2060, "time": 59.1},
        0.4: {"val_loss": 0.4822, "val_psnr": 12.16, "val_ssim": 0.3310, "val_mae": 0.2021, "time": 60.7},
        0.5: {"val_loss": 0.4443, "val_psnr": 11.80, "val_ssim": 0.3249, "val_mae": 0.2136, "time": 66.0},
        0.6: {"val_loss": 0.3841, "val_psnr": 12.67, "val_ssim": 0.3289, "val_mae": 0.1928, "time": 61.8},
        0.7: {"val_loss": 0.3038, "val_psnr": 13.86, "val_ssim": 0.3695, "val_mae": 0.1638, "time": 60.3},
        0.8: {"val_loss": 0.2770, "val_psnr": 12.67, "val_ssim": 0.3661, "val_mae": 0.1877, "time": 59.0},
        0.9: {"val_loss": 0.2534, "val_psnr": 11.14, "val_ssim": 0.3728, "val_mae": 0.2118, "time": 60.1},
        1.0: {"val_loss": 0.1103, "val_psnr": 16.02, "val_ssim": 0.3921, "val_mae": 0.1103, "time": 60.2},
    }

    # 1. Log individual runs
    for alpha, metrics in val_results.items():
        run = tracker.start_run(run_name=f"alpha_{alpha:.1f}", experiment_name=exp_name)
        tracker.log_params({
            "alpha": alpha,
            "loss_formulation": f"{alpha:.1f}*L1 + {1.0-alpha:.1f}*(1-SSIM)",
            "epochs": 5,
            "architecture": "ResBlockAutoencoder",
            "channels": "(16, 32, 64, 128)",
            "bottleneck_dim": 128,
            "dataset": "Oxford-IIIT Pet (15% subset)",
        })

        for ep, tr_loss in enumerate(epoch_train_losses[alpha], start=1):
            tracker.log_metric("train_loss", tr_loss, step=ep)

        tracker.log_metrics({
            "val_loss": metrics["val_loss"],
            "val_psnr": metrics["val_psnr"],
            "val_ssim": metrics["val_ssim"],
            "val_mae": metrics["val_mae"],
            "elapsed_seconds": metrics["time"],
        }, step=5)
        tracker.end_run()

    # 2. Generate and log summary comparison plot
    alphas = sorted(val_results.keys())
    val_losses = [val_results[a]["val_loss"] for a in alphas]
    val_ssims = [val_results[a]["val_ssim"] for a in alphas]
    val_psnrs = [val_results[a]["val_psnr"] for a in alphas]
    val_maes = [val_results[a]["val_mae"] for a in alphas]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(r"Task 1: Loss Function Sensitivity Analysis ($\alpha \in [0.0, 1.0]$)", fontsize=14, fontweight="bold")

    # Panel 1: Loss
    axes[0, 0].plot(alphas, val_losses, "o-", color="#d62728", lw=2, markersize=6)
    axes[0, 0].set_title("Validation Loss (Combined)", fontweight="bold")
    axes[0, 0].set_xlabel(r"$\alpha$ (Weight on L1)")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: SSIM
    axes[0, 1].plot(alphas, val_ssims, "s-", color="#9467bd", lw=2, markersize=6)
    axes[0, 1].axvline(0.8, color="#2ca02c", linestyle="--", label=r"Recommended $\alpha=0.8$")
    axes[0, 1].set_title("Validation SSIM (Structural Fidelity)", fontweight="bold")
    axes[0, 1].set_xlabel(r"$\alpha$ (Weight on L1)")
    axes[0, 1].set_ylabel("SSIM")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    # Panel 3: PSNR
    axes[1, 0].plot(alphas, val_psnrs, "^-", color="#2ca02c", lw=2, markersize=6)
    axes[1, 0].set_title("Validation PSNR", fontweight="bold")
    axes[1, 0].set_xlabel(r"$\alpha$ (Weight on L1)")
    axes[1, 0].set_ylabel("PSNR (dB)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: MAE
    axes[1, 1].plot(alphas, val_maes, "d-", color="#ff7f0e", lw=2, markersize=6)
    axes[1, 1].set_title("Validation MAE (Pixel Error)", fontweight="bold")
    axes[1, 1].set_xlabel(r"$\alpha$ (Weight on L1)")
    axes[1, 1].set_ylabel("Mean Absolute Error")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    out_dir = Path("results/task1/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "alpha_spectrum_analysis.png"
    fig.savefig(fig_path, dpi=200)

    # Log summary run with figure
    tracker.start_run(run_name="alpha_spectrum_summary", experiment_name=exp_name)
    tracker.log_params({"spectrum": "0.0 to 1.0 (step 0.1)", "best_ssim_alpha": 1.0, "recommended_alpha": 0.8})
    tracker.log_figure(fig, artifact_file="alpha_spectrum_analysis.png")
    tracker.end_run()

    print(f"Successfully logged 11 alpha runs + summary figure to MLflow under '{exp_name}'!", flush=True)
    print(f"Saved figure to: {fig_path}", flush=True)


if __name__ == "__main__":
    sync_loss_research_to_mlflow()
