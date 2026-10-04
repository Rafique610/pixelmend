"""Measure parameter count, file size and CPU latency of every production ONNX model.

Run: uv run python scripts/benchmark_cpu.py
Writes results/app/cpu_benchmark.json (CPUExecutionProvider only, batch=1, 128x128).
"""

import json
import os
import platform
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

MODELS_DIR = "models/onnx"
WARMUP, RUNS = 10, 50


def count_params(path: str) -> int:
    model = onnx.load(path, load_external_data=True)
    return int(sum(int(np.prod(numpy_helper.to_array(t).shape)) for t in model.graph.initializer))


def make_inputs(sess: ort.InferenceSession) -> dict:
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if "int" in inp.type:
            feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            feeds[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feeds


def main() -> None:
    out = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "warmup_runs": WARMUP,
        "timed_runs": RUNS,
        "models": {},
    }
    for name in sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")):
        path = os.path.join(MODELS_DIR, name)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feeds = make_inputs(sess)
        for _ in range(WARMUP):
            sess.run(None, feeds)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t0) * 1000)
        out["models"][name] = {
            "params": count_params(path),
            "size_mb": round(os.path.getsize(path) / 1048576, 2),
            "latency_ms_mean": round(statistics.mean(times), 2),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 2),
        }
        print(name, out["models"][name])
    os.makedirs("results/app", exist_ok=True)
    with open("results/app/cpu_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
