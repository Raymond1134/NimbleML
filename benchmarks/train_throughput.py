#!/usr/bin/env python3
"""GPT training throughput: forward + backward + AdamW (+ grad clip).

Compare NimbleML against a PyTorch reference model with the same pinned config.

Usage:
  python benchmarks/train_throughput.py
  python benchmarks/train_throughput.py --quick
  python benchmarks/train_throughput.py --cpu
  python benchmarks/train_throughput.py --no-torch
  python benchmarks/train_throughput.py --json results.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_early_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", action="store_true")
    args, _ = parser.parse_known_args()
    return args


_early = _parse_early_args()
if _early.cpu:
    os.environ["NIMBLEML_DEVICE"] = "cpu"
else:
    os.environ.setdefault("NIMBLEML_DEVICE", "gpu")

from benchmarks._nimble_train import build_train_step, make_inputs, make_model
from benchmarks._timing import (
    bench,
    format_row,
    peak_vram_mb,
    print_header,
    reset_peak_vram_mb,
)
from benchmarks._torch_ref import build_torch_gpt
from benchmarks.config import QUICK, REFERENCE, ReferenceConfig


def _resolve_config(quick: bool, warmup: int | None, runs: int | None) -> ReferenceConfig:
    base = QUICK if quick else REFERENCE
    if warmup is None and runs is None:
        return base
    return ReferenceConfig(
        vocab=base.vocab,
        d_model=base.d_model,
        heads=base.heads,
        layers=base.layers,
        ff_mult=base.ff_mult,
        seq=base.seq,
        batch=base.batch,
        warmup=base.warmup if warmup is None else warmup,
        runs=base.runs if runs is None else runs,
        lr=base.lr,
        weight_decay=base.weight_decay,
        max_grad_norm=base.max_grad_norm,
    )


def _bench_with_vram(name, fn, *, np_module, using_gpu, cfg, tokens):
    reset_peak_vram_mb(np_module, using_gpu)
    result = bench(
        name,
        fn,
        np_module=np_module,
        using_gpu=using_gpu,
        warmup=cfg.warmup,
        runs=cfg.runs,
        tokens=tokens,
    )
    vram = peak_vram_mb(np_module, using_gpu)
    if vram is not None:
        result["peak_vram_mb"] = vram
    return result


def run_nimble(cfg: ReferenceConfig) -> dict:
    from NimbleML.utils.np_backend import device, dtype, np, set_dtype, using_gpu

    set_dtype("float32")
    model = make_model(cfg)
    inputs, targets, tokens = make_inputs(cfg)
    train_step = build_train_step(model, inputs, targets, cfg)
    result = _bench_with_vram(
        "nimble_train_step",
        train_step,
        np_module=np,
        using_gpu=using_gpu,
        cfg=cfg,
        tokens=tokens,
    )
    result["device"] = "gpu" if using_gpu else "cpu"
    result["backend_device"] = device
    result["dtype"] = str(dtype)
    result["using_gpu"] = using_gpu
    return result


def run_torch(cfg: ReferenceConfig, *, nimble_on_gpu: bool) -> dict | None:
    ref = build_torch_gpt(cfg, nimble_on_gpu=nimble_on_gpu)
    if ref is None:
        return None
    if "error" in ref:
        return ref

    device = ref["device"]
    train_step = ref["train_step"]
    sync = ref["sync"]
    reset_vram = ref["reset_vram"]
    peak_vram = ref["peak_vram_mb"]
    tokens = ref["tokens"]

    for _ in range(cfg.warmup):
        train_step()
    sync()

    reset_vram()
    timings_ms = []
    for _ in range(cfg.runs):
        t0 = time.perf_counter()
        train_step()
        sync()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    mean_ms = statistics.mean(timings_ms)
    result = {
        "name": "torch_train_step",
        "mean_ms": mean_ms,
        "p50_ms": timings_ms[len(timings_ms) // 2],
        "p95_ms": timings_ms[int(0.95 * (len(timings_ms) - 1))],
        "tokens": tokens,
        "tokens_per_sec": tokens / (mean_ms / 1000.0),
        "device": str(device),
        "torch_version": ref.get("torch_version"),
    }
    vram = peak_vram()
    if vram is not None:
        result["peak_vram_mb"] = vram
    return result


def _devices_match(nimble: dict, torch_result: dict | None) -> bool:
    if torch_result is None or "error" in torch_result:
        return False
    nimble_on_gpu = bool(nimble.get("using_gpu"))
    torch_dev = str(torch_result.get("device", "")).lower()
    torch_on_gpu = "cuda" in torch_dev
    return nimble_on_gpu == torch_on_gpu


def _print_ratio(nimble: dict, torch_result: dict | None) -> None:
    if torch_result is None or "tokens_per_sec" not in nimble:
        return
    if "error" in torch_result:
        print("-" * 96)
        print(f"WARNING: {torch_result['message']}")
        if torch_result.get("torch_version"):
            print(f"  Installed torch: {torch_result['torch_version']} (cuda={torch_result.get('torch_cuda')})")
        return
    if not _devices_match(nimble, torch_result):
        print("-" * 96)
        print("WARNING: Skipping ratio — NimbleML and PyTorch ran on different devices.")
        return
    ratio = torch_result["tokens_per_sec"] / nimble["tokens_per_sec"]
    print("-" * 96)
    print(
        f"PyTorch / NimbleML train throughput ratio: {ratio:.2f}x "
        f"(1.0x = same speed; >1.0 = PyTorch faster)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT train-step throughput benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use smaller QUICK config from benchmarks/config.py.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend for both NimbleML and PyTorch.")
    parser.add_argument("--no-torch", action="store_true", help="Skip PyTorch comparison.")
    parser.add_argument("--json", type=str, default="", help="Write results JSON to path.")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()
    cfg = _resolve_config(args.quick, args.warmup, args.runs)

    from NimbleML.utils.np_backend import dtype

    nimble = run_nimble(cfg)
    torch_result = None if args.no_torch else run_torch(cfg, nimble_on_gpu=nimble.get("using_gpu", False))

    print_header("GPT train throughput", cfg, dtype=str(dtype))
    print(format_row(nimble))
    if torch_result is not None and "error" not in torch_result:
        print(format_row(torch_result))
    elif torch_result is not None and "error" in torch_result:
        print("torch_train_step [skipped]     (device mismatch — see warning below)")
    else:
        print("PyTorch comparison skipped (install torch or remove --no-torch).")
    _print_ratio(nimble, torch_result)

    payload = {"config": cfg.__dict__, "nimble": nimble, "torch": torch_result}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
