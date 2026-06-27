"""Shared timing helpers for benchmarks."""
from __future__ import annotations

import statistics
import time
from typing import Callable

BenchFn = Callable[[], None]


def sync_backend(np_module, using_gpu: bool) -> None:
    if using_gpu and hasattr(np_module, "cuda"):
        np_module.cuda.Stream.null.synchronize()


def reset_peak_vram_mb(np_module, using_gpu: bool) -> None:
    if not using_gpu or not hasattr(np_module, "cuda"):
        return
    try:
        pool = np_module.cuda.get_default_memory_pool()
        if hasattr(pool, "reset_peak_stats"):
            pool.reset_peak_stats()
    except Exception:
        pass


def peak_vram_mb(np_module, using_gpu: bool) -> float | None:
    if not using_gpu or not hasattr(np_module, "cuda"):
        return None
    try:
        pool = np_module.cuda.get_default_memory_pool()
        if hasattr(pool, "peak_bytes"):
            return pool.peak_bytes() / (1024 * 1024)
        return pool.used_bytes() / (1024 * 1024)
    except Exception:
        return None


def bench(
    name: str,
    fn: BenchFn,
    *,
    np_module,
    using_gpu: bool,
    warmup: int,
    runs: int,
    tokens: float | None = None,
) -> dict:
    for _ in range(warmup):
        fn()
    sync_backend(np_module, using_gpu)

    timings_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync_backend(np_module, using_gpu)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    mean_ms = statistics.mean(timings_ms)
    p50_ms = timings_ms[len(timings_ms) // 2]
    p95_ms = timings_ms[int(0.95 * (len(timings_ms) - 1))]
    out = {"name": name, "mean_ms": mean_ms, "p50_ms": p50_ms, "p95_ms": p95_ms}
    if tokens is not None and mean_ms > 0:
        out["tokens"] = tokens
        out["tokens_per_sec"] = tokens / (mean_ms / 1000.0)
    return out


def _short_device(device) -> str:
    text = str(device).lower()
    if "cuda" in text or text == "gpu":
        return "gpu"
    if text == "cpu":
        return "cpu"
    return text


def format_row(result: dict) -> str:
    tps = f"{result['tokens_per_sec']:,.0f}" if "tokens_per_sec" in result else "-"
    vram = f"{result['peak_vram_mb']:.0f} MB" if "peak_vram_mb" in result else "-"
    dev = _short_device(result.get("device", ""))
    name = f"{result['name']} [{dev}]" if dev else result["name"]
    return (
        f"{name:28} {result['mean_ms']:10.3f} {result['p50_ms']:10.3f} "
        f"{result['p95_ms']:10.3f} {tps:>12} {vram:>10}"
    )


def print_header(title: str, cfg, *, dtype: str, device: str | None = None) -> None:
    label = f" ({device}, dtype={dtype})" if device else f" (dtype={dtype})"
    print(f"\n{title}{label}")
    print(
        f"config: vocab={cfg.vocab} d_model={cfg.d_model} layers={cfg.layers} "
        f"heads={cfg.heads} ff_mult={cfg.ff_mult} batch={cfg.batch} seq={cfg.seq} "
        f"(warmup={cfg.warmup} runs={cfg.runs})"
    )
    print("-" * 96)
    print(f"{'name':28} {'mean ms':>10} {'p50 ms':>10} {'p95 ms':>10} {'tok/s':>12} {'peak VRAM':>10}")
    print("-" * 96)
