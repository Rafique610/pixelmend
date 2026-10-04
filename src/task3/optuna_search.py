"""src/task3/optuna_search.py: Optuna HPO Study and Retrain for Task 3 Soft MoE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as ovm
import pytorch_msssim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from src.shared.config import Settings, get_settings
from src.shared.corruptions import apply_corruption, sample_random_corruption
from src.shared.datasets.corrupted import get_corrupted_pet_dataloader
from src.shared.datasets.pets import PetDataset
from src.shared.manifests import load_val_manifest
from src.shared.optuna_utils import create_or_load_study, make_tracking_callback
from src.shared.tracking import ExperimentTracker
from src.task3.moe_model import SoftMoE, build_soft_moe
from src.task3.train import train_two_stage
from src.task3.training_utils import evaluate_validation, train_one_epoch

CORRUPTION_NAMES = {0: "clean", 1: "salt_and_pepper", 2: "gaussian_blur", 3: "occlusion"}


def check_routing_collapse(w: Sequence[float]) -> bool:
    return bool(w) and (max(w) > 0.90 or min(w) < 0.02)


def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "fine_tune_lr": trial.suggest_float("fine_tune_lr", 1e-5, 1e-3, log=True),
        "temperature": trial.suggest_float("temperature", 0.1, 5.0),
        "lambda_ce": trial.suggest_float("lambda_ce", 0.01, 0.5, log=True),
        "lambda_balance": trial.suggest_float("lambda_balance", 0.001, 0.1, log=True),
        "reconstruction_alpha": trial.suggest_float("reconstruction_alpha", 0.5, 1.0),
    }


def compute_derived_lambdas(params: Dict[str, float]) -> Tuple[float, float, float, float]:
    alpha = params["reconstruction_alpha"]
    return (alpha, 1.0 - alpha, params["lambda_ce"], params["lambda_balance"])


def load_moe(dev: torch.device) -> SoftMoE:
    return build_soft_moe(
        "checkpoints/task2/classifier_best.pt",
        "checkpoints/task2/specialist_salt_best.pt",
        "checkpoints/task2/specialist_blur_best.pt",
        "checkpoints/task2/specialist_occlusion_best.pt",
    ).to(dev)


def evaluate_per_corruption(
    model: nn.Module, loader: DataLoader, dev: torch.device, tau: float = 1.0
) -> Dict[str, Any]:
    val = evaluate_validation(model, loader, dev, (0.8, 0.2, 0.1, 0.01), "l2_deviation", tau)
    p_data: Dict[int, Dict[str, List[float]]] = {c: {"p": [], "s": [], "m": []} for c in range(4)}
    p_w: Dict[int, List[torch.Tensor]] = {c: [] for c in range(4)}
    with torch.no_grad():
        for corr, clean, lbl in loader:
            corr, clean, lbl = corr.to(dev), clean.to(dev), lbl.to(dev)
            rec, w, _ = model(corr, tau=tau)
            mse = torch.mean((rec - clean) ** 2, dim=[1, 2, 3])
            psnr = (10.0 * torch.log10(1.0 / (mse + 1e-8))).cpu().tolist()
            ssim = pytorch_msssim.ssim(
                rec, clean, data_range=1.0, size_average=False
            ).cpu().tolist()
            mae = torch.mean(torch.abs(rec - clean), dim=[1, 2, 3]).cpu().tolist()
            for i, c in enumerate(lbl.cpu().tolist()):
                p_data[c]["p"].append(psnr[i])
                p_data[c]["s"].append(ssim[i])
                p_data[c]["m"].append(mae[i])
                p_w[c].append(w[i : i + 1].cpu())

    val["per_corruption"] = {
        CORRUPTION_NAMES[c]: {
            "val_psnr": round(float(sum(p_data[c]["p"]) / max(1, len(p_data[c]["p"]))), 2),
            "val_ssim": round(float(sum(p_data[c]["s"]) / max(1, len(p_data[c]["s"]))), 4),
            "val_mae": round(float(sum(p_data[c]["m"]) / max(1, len(p_data[c]["m"]))), 4),
            "avg_routing_vector": [
                round(x, 4) for x in (torch.cat(p_w[c], 0).mean(0).tolist() if p_w[c] else [0.0]*4)
            ],
        } for c in range(4)
    }
    return val


def build_fast_optuna_loaders(
    batch_size: int = 32, num_train: int = 1024, seed: int = 42, settings: Optional[Settings] = None
) -> Tuple[DataLoader, DataLoader]:
    cfg = settings or get_settings()
    b_tr = PetDataset(split="train", image_size=128, return_labels=False, settings=cfg)
    b_val = PetDataset(split="val", image_size=128, return_labels=False, settings=cfg)
    v_cr, v_cl, v_lb = [], [], []
    for it in load_val_manifest(settings=cfg):
        c = b_val[it["val_id"]]
        v_cl.append(c)
        v_cr.append(apply_corruption(c, it["corruption_type"], it["params"]))
        v_lb.append(it["corruption_label"])
    val_loader = DataLoader(
        TensorDataset(torch.stack(v_cr), torch.stack(v_cl), torch.tensor(v_lb)),
        batch_size=batch_size, shuffle=False,
    )
    torch.manual_seed(seed)
    spc, counts, t_cr, t_cl, t_lb = num_train // 4, {c: 0 for c in range(4)}, [], [], []
    for idx in torch.randperm(len(b_tr)).tolist():
        if all(counts[c] >= spc for c in range(4)):
            break
        for c in range(4):
            if counts[c] < spc:
                c_name = CORRUPTION_NAMES[c]
                _, _, pm = sample_random_corruption(corruption_type=c_name, image_size=128)
                t_cl.append(b_tr[idx])
                t_cr.append(apply_corruption(b_tr[idx], c_name, pm))
                t_lb.append(c)
                counts[c] += 1
    train_loader = DataLoader(
        TensorDataset(torch.stack(t_cr), torch.stack(t_cl), torch.tensor(t_lb)),
        batch_size=batch_size, shuffle=True,
    )
    return train_loader, val_loader


def create_moe_objective(
    train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
    warmup_epochs: int = 1, joint_epochs: int = 3,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        lambdas = compute_derived_lambdas(params)
        tau = params["temperature"]
        model = load_moe(device)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        best_ssim, ep = -1.0, 0

        def _step(phase: str, opt: torch.optim.Optimizer) -> None:
            nonlocal ep, best_ssim
            ep += 1
            train_one_epoch(
                model, train_loader, opt, scaler, device, phase, lambdas, "l2_deviation", tau
            )
            val = evaluate_validation(model, val_loader, device, lambdas, "l2_deviation", tau)
            if check_routing_collapse(val["avg_routing_vector"]):
                raise optuna.TrialPruned("Routing collapse detected")
            best_ssim = max(best_ssim, val["val_ssim"])
            trial.report(val["val_ssim"], step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        opt_p1 = torch.optim.Adam(
            [p for p in model.gate.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
        )
        for _ in range(warmup_epochs):
            _step("warmup", opt_p1)

        model.set_experts_frozen(False)
        specs = (model.specialist_salt, model.specialist_blur, model.specialist_occlusion)
        opt_p2 = torch.optim.Adam([
            {"params": [p for p in model.gate.parameters() if p.requires_grad],
             "lr": params["fine_tune_lr"], "weight_decay": 1e-4},
            {"params": [p for s in specs for p in s.parameters() if p.requires_grad],
             "lr": params["fine_tune_lr"] * 0.2, "weight_decay": 1e-4},
        ])
        for _ in range(joint_epochs):
            _step("joint", opt_p2)

        return float(best_ssim)

    return objective


def export_best_params(
    study: optuna.Study, output_path: Union[str, Path] = "config/task3_best_params.json"
) -> Dict[str, Any]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    b = study.best_trial
    l1, ssim, ce, bal = compute_derived_lambdas(b.params)
    data = {
        "study_name": study.study_name, "best_trial_number": b.number,
        "best_val_ssim": round(float(b.value), 4), "best_params": b.params,
        "derived_lambdas": {
            "lambda_l1": round(l1, 4), "lambda_ssim": round(ssim, 4),
            "lambda_ce": round(ce, 4), "lambda_bal": round(bal, 4),
        },
        "total_trials": len(study.trials),
        "completed_trials": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
        "pruned_trials": sum(1 for t in study.trials if t.state.name == "PRUNED"),
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def generate_optuna_plots(
    study: optuna.Study, output_dir: Union[str, Path] = "results/task3/figures"
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, fn in [
        ("optuna_moe_history.png", ovm.plot_optimization_history),
        ("optuna_moe_param_importances.png", ovm.plot_param_importances),
        ("optuna_moe_slice.png", ovm.plot_slice),
    ]:
        try:
            ax = fn(study)
            if hasattr(ax, "figure"):
                fig = ax.figure
            elif hasattr(ax, "flat"):
                fig = ax.flat[0].figure
            elif hasattr(ax, "__iter__"):
                fig = list(ax)[0].figure
            else:
                fig = plt.gcf()
            fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"Warning plotting {fname}: {e}")
        finally:
            plt.close("all")


def execute_definitive_retrain(
    params_path: Union[str, Path] = "config/task3_best_params.json",
    warmup_epochs: int = 3, joint_epochs: int = 8, seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)
    params = best_cfg["best_params"]
    lambdas = compute_derived_lambdas(params)
    tau, ft_lr = params["temperature"], params["fine_tune_lr"]
    dev = device or get_settings().torch_device
    print(f"\n=== Definitive Retrain on {dev} ===\nParams: lr={ft_lr:.2e}, tau={tau:.2f}")
    cfg = get_settings()
    train_loader = get_corrupted_pet_dataloader("train", batch_size=32, num_workers=2, settings=cfg)
    val_loader = get_corrupted_pet_dataloader("val", batch_size=32, num_workers=2, settings=cfg)
    args = argparse.Namespace(
        warmup_epochs=warmup_epochs, joint_epochs=joint_epochs, batch_size=32, warmup_lr=1e-3,
        gate_lr=ft_lr, specialist_lr=ft_lr * 0.2, weight_decay=1e-4, tau=tau,
        lambda_l1=lambdas[0], lambda_ssim=lambdas[1], lambda_ce=lambdas[2], lambda_bal=lambdas[3],
        balance_variant="l2_deviation", save_path="checkpoints/task3/best_model.pth",
        metrics_path="results/task3/final_train_metrics.json", run_name="task3-moe-final",
        device=str(dev), num_workers=2, seed=seed, smoke_test=False,
    )
    trained_model, summary = train_two_stage(load_moe(dev), train_loader, val_loader, args, dev)
    val_metrics = evaluate_per_corruption(trained_model, val_loader, dev, tau=tau)
    val_metrics["best_epoch"] = summary["best_epoch"]
    val_metrics["hyperparameters"] = params
    res_path = Path("results/task3/final_val_metrics.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    print(f"Retrain finished! Checkpoint: {args.save_path} | Metrics: {res_path}")
    return val_metrics


def run_moe_optuna_study(
    n_trials: int = 25, warmup_epochs: int = 1, joint_epochs: int = 3, seed: int = 42
) -> optuna.Study:
    cfg, dev = get_settings(), get_settings().torch_device
    print(f"\n=== Starting Task 3 Soft MoE Optuna Study ({n_trials} trials on {dev}) ===")
    train_loader, val_loader = build_fast_optuna_loaders(
        batch_size=32, num_train=1024, seed=seed, settings=cfg
    )
    study = create_or_load_study(
        "task3-moe-joint", direction="maximize", pruner_name="median", warmup_steps=3, seed=seed
    )
    cb = make_tracking_callback(
        tracker=ExperimentTracker(settings=cfg),
        experiment_name="genai-task3-soft-moe",
        metric_name="val_ssim",
    )
    obj = create_moe_objective(
        train_loader, val_loader, dev, warmup_epochs=warmup_epochs, joint_epochs=joint_epochs
    )
    study.optimize(obj, n_trials=n_trials, callbacks=[cb])
    export_best_params(study, "config/task3_best_params.json")
    generate_optuna_plots(study, "results/task3/figures")
    print(f"\nOptuna complete! Best trial #{study.best_trial.number}: SSIM={study.best_value:.4f}")
    return study

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 Soft MoE Optuna Study and Retrain")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--joint-epochs", type=int, default=8)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    if a.retrain:
        execute_definitive_retrain(warmup_epochs=a.warmup_epochs, joint_epochs=a.joint_epochs, seed=a.seed)
    else:
        run_moe_optuna_study(a.n_trials, a.warmup_epochs, a.joint_epochs, a.seed)

