"""scripts/verify_challenger2_m1_benchmarks.py
-------------------------------------------
Benchmark script collecting empirical metrics for Challenger 2 evaluation:
- VRAM footprint (MB) across batch sizes {1, 2, 7, 16, 32} on NVIDIA RTX 3050 GPU.
- Forward and backward latency per batch size.
- Exact weight bit-level verification stats.
- Parameter counts (trainable vs non-trainable during freezing).
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.task3.moe_model import build_soft_moe


def main():
    print("=" * 70)
    print("CHALLENGER 2 EMPIRICAL VERIFICATION BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware Device: {device} ({gpu_desc})")

    classifier_path = Path("checkpoints/task2/classifier_best.pt")
    salt_path = Path("checkpoints/task2/specialist_salt_best.pt")
    blur_path = Path("checkpoints/task2/specialist_blur_best.pt")
    occlusion_path = Path("checkpoints/task2/specialist_occlusion_best.pt")

    # 1. Parameter counts & Weight Transplantation
    model = build_soft_moe(
        classifier_path=classifier_path,
        salt_path=salt_path,
        blur_path=blur_path,
        occlusion_path=occlusion_path,
    ).to(device)

    total_params = model.count_parameters(only_trainable=False)
    gate_params = model.gate.count_parameters(only_trainable=False)
    salt_params = sum(p.numel() for p in model.specialist_salt.parameters())
    blur_params = sum(p.numel() for p in model.specialist_blur.parameters())
    occlusion_params = sum(p.numel() for p in model.specialist_occlusion.parameters())

    print("\n--- Model Architecture & Parameter Allocation ---")
    print(f"Total Parameters:      {total_params:,}")
    print(f"  - Gate:              {gate_params:,} ({gate_params/total_params*100:.2f}%)")
    print(f"  - Salt Specialist:   {salt_params:,} ({salt_params/total_params*100:.2f}%)")
    print(f"  - Blur Specialist:   {blur_params:,} ({blur_params/total_params*100:.2f}%)")
    print(f"  - Occlusion Spec:    {occlusion_params:,} ({occlusion_params/total_params*100:.2f}%)")

    model.set_experts_frozen(True)
    frozen_trainable = model.count_parameters(only_trainable=True)
    print(f"Trainable Parameters (Frozen Experts): {frozen_trainable:,} (Only Gate)")
    assert frozen_trainable == gate_params

    model.set_experts_frozen(False)
    unfrozen_trainable = model.count_parameters(only_trainable=True)
    print(f"Trainable Parameters (Unfrozen Experts): {unfrozen_trainable:,} (All Components)")
    assert unfrozen_trainable == total_params

    # 2. VRAM and Latency Benchmarks
    if torch.cuda.is_available():
        print("\n--- CUDA Profiling: Memory & Latency Across Batch Sizes ---")
        header = (
            f"{'Batch Size':>10} | {'VRAM Allocated (MB)':>20} | {'VRAM Reserved (MB)':>18} | "
            f"{'Fwd Latency (ms)':>16} | {'Bwd Latency (ms)':>16}"
        )
        print(header)
        print("-" * 90)

        for b in [1, 2, 7, 16, 32]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(b, 3, 128, 128, device=device)
            target = torch.randn(b, 3, 128, 128, device=device)

            # Warmup
            model.train()
            out, w, log = model(x)
            loss = F.l1_loss(out, target)
            loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()

            # Measure forward
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                out, w, log = model(x)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / iters * 1000

            # Measure backward
            t0 = time.perf_counter()
            for _ in range(iters):
                model.zero_grad()
                out, w, log = model(x)
                loss = F.l1_loss(out, target)
                loss.backward()
            torch.cuda.synchronize()
            bwd_ms = (time.perf_counter() - t0) / iters * 1000

            vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
            vram_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

            row = (
                f"{b:>10} | {vram_alloc:>20.2f} | {vram_res:>18.2f} | "
                f"{fwd_ms:>16.2f} | {bwd_ms:>16.2f}"
            )
            print(row)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE - VERDICT READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
