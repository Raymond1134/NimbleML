"""Performance micro-benchmarks for NimbleML core ops.

Baseline (Jun 16, 2026; local machine)
---------------------------------------
CPU:
- Dense forward+backward: 2.367 ms/op
- Conv2D forward+backward: 46.917 ms/op
- Embedding lookup forward: 1.823 ms/op
- Attention forward: 8.061 ms/op

GPU:
- Dense forward+backward: 1.288 ms/op
- Conv2D forward+backward: 45.465 ms/op
- Embedding lookup forward: 0.796 ms/op
- Attention forward: 1.388 ms/op

Top hotspots from this baseline:
1) Conv2D fwd+bwd (dominated by im2col/col2im-style memory movement + Python overhead)
2) CPU attention forward (QK^T and softmax path, heavy temporary tensors)
3) Dense fwd+bwd at medium batch sizes (autograd bookkeeping and frequent array reshapes)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NimbleML.layers import Embedding
from NimbleML.layers.conv2d import Conv2D
from NimbleML.layers.dense import Dense
from NimbleML.neural_network.attention import Attention, make_causal_mask
from NimbleML.utils.np_backend import np, set_dtype, using_gpu
from NimbleML.utils.tensor import Tensor


def _sync_if_gpu() -> None:
    if using_gpu and hasattr(np, "cuda"):
        np.cuda.Stream.null.synchronize()


def _bench(name: str, fn, warmup: int = 8, runs: int = 30) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync_if_gpu()

    timings_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync_if_gpu()
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    timings_ms.sort()
    mean_ms = statistics.mean(timings_ms)
    p50_ms = timings_ms[len(timings_ms) // 2]
    p95_ms = timings_ms[int(0.95 * (len(timings_ms) - 1))]
    return {"name": name, "mean_ms": mean_ms, "p50_ms": p50_ms, "p95_ms": p95_ms}


def _dense_case() -> dict[str, float]:
    batch, in_dim, out_dim = 128, 512, 512
    layer = Dense(in_dim, out_dim)
    x = Tensor(np.random.standard_normal((batch, in_dim)).astype(np.float32).ravel(), (batch, in_dim), requires_grad=True)

    def step():
        x.grad = None
        layer.weights.grad = None
        layer.biases.grad = None
        out = layer.forward(x)
        out.sum().backward()

    result = _bench("dense_fwd_bwd", step)
    result["elements"] = float(batch * in_dim)
    return result


def _conv2d_case() -> dict[str, float]:
    n, c, h, w = 8, 32, 32, 32
    layer = Conv2D(c, 64, kernel_size=3, stride=1, padding=1, bias=True)
    x_arr = np.random.standard_normal((n, c, h, w)).astype(np.float32)
    x = Tensor(x_arr.ravel(), (n, c, h, w), requires_grad=True)

    def step():
        x.grad = None
        layer.weights.grad = None
        layer.biases.grad = None
        out = layer.forward(x)
        out.sum().backward()

    result = _bench("conv2d_fwd_bwd", step)
    result["elements"] = float(n * c * h * w)
    return result


def _embedding_case() -> dict[str, float]:
    vocab_size, embed_dim = 50_000, 256
    batch, seq = 32, 128
    layer = Embedding(vocab_size=vocab_size, embed_dim=embed_dim)
    ids = np.random.randint(0, vocab_size, size=(batch, seq)).tolist()

    def step():
        _ = layer.forward(ids)

    result = _bench("embedding_lookup_fwd", step)
    result["tokens"] = float(batch * seq)
    return result


def _attention_case() -> dict[str, float]:
    batch, seq_len, d_k = 8, 256, 128
    q = Tensor(np.random.standard_normal((batch, seq_len, d_k)).astype(np.float32).ravel(), (batch, seq_len, d_k))
    k = Tensor(np.random.standard_normal((batch, seq_len, d_k)).astype(np.float32).ravel(), (batch, seq_len, d_k))
    v = Tensor(np.random.standard_normal((batch, seq_len, d_k)).astype(np.float32).ravel(), (batch, seq_len, d_k))
    mask = make_causal_mask(seq_len)
    attn = Attention(d_k)

    def step():
        _ = attn.forward(q, k, v, mask=mask)

    result = _bench("attention_fwd", step)
    result["tokens"] = float(batch * seq_len)
    return result


def run_once() -> list[dict[str, float]]:
    set_dtype("float32")
    return [_dense_case(), _conv2d_case(), _embedding_case(), _attention_case()]


def print_report(results: list[dict[str, float]]) -> None:
    device_name = "GPU" if using_gpu else "CPU"
    print(f"\nNimbleML micro-benchmark report ({device_name})")
    print("-" * 72)
    print(f"{'op':28} {'mean ms/op':>12} {'p50 ms':>10} {'p95 ms':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:28} {r['mean_ms']:12.3f} {r['p50_ms']:10.3f} {r['p95_ms']:10.3f}")
    print("-" * 72)


def run_both_devices() -> None:
    base_env = os.environ.copy()
    for device in ("cpu", "gpu"):
        env = dict(base_env)
        env["NIMBLEML_DEVICE"] = device
        print(f"\n=== Running benchmarks on {device.upper()} ===")
        completed = subprocess.run([sys.executable, __file__], env=env, check=False)
        if completed.returncode != 0:
            print(f"Skipping {device.upper()} (backend unavailable or run failed).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NimbleML performance benchmarks.")
    parser.add_argument("--both", action="store_true", help="Run benchmark subprocesses on CPU and GPU.")
    args = parser.parse_args()

    if args.both:
        run_both_devices()
        return

    results = run_once()
    print_report(results)


if __name__ == "__main__":
    main()
