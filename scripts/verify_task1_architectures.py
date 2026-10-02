"""Empirical benchmark comparing Task 1 Autoencoder architectural variants.

Measures parameter counts, compression ratios, forward latency, and memory footprints.
"""

import time
import torch

from src.task1.autoencoder import UniversalAutoencoder


def benchmark_variants():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    variants = [
        ("Option 1: Plain Conv Stack (Strict Bottleneck)", False, False),
        ("Option 2: ResBlock-based AE (No Skips)", True, False),
        ("Option 3: U-Net-lite (Restricted Skips)", False, True),
        ("Option 4: ResBlock + U-Net-lite", True, True),
    ]

    x = torch.rand(16, 3, 128, 128, device=device)

    print("\n" + "=" * 90)
    print(f"{'Variant':<45} | {'Params':<10} | {'Ratio':<6} | {'Latency (ms)':<12} | {'Output Shape'}")
    print("=" * 90)

    results = []
    for name, use_res, use_skips in variants:
        model = UniversalAutoencoder(
            in_channels=3,
            out_channels=3,
            channels=(32, 64, 128, 256),
            bottleneck_dim=256,
            use_residual=use_res,
            use_skips=use_skips,
        ).to(device)

        params = model.count_parameters()
        ratio = model.compression_ratio((3, 128, 128))

        # Warmup
        for _ in range(5):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timing
        start = time.perf_counter()
        iters = 20
        for _ in range(iters):
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = ((time.perf_counter() - start) / iters) * 1000

        print(f"{name:<45} | {params:<10,d} | {ratio:<6.1f}x | {latency:<12.2f} | {list(out.shape)}")
        results.append((name, params, ratio, latency))

    print("=" * 90)
    return results


if __name__ == "__main__":
    benchmark_variants()
