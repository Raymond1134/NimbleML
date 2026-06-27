#!/usr/bin/env python3
"""Forward-only benchmarks: isolate compute vs Python autograd overhead.

Compares raw NumPy/CuPy attention math, NimbleML modules, and full GPT forward.

Usage:
  python benchmarks/forward_only.py
  python benchmarks/forward_only.py --quick
  python benchmarks/forward_only.py --cpu
"""
from __future__ import annotations

import argparse
import os
import sys
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

from benchmarks._nimble_train import make_inputs, make_model, zero_grads
from benchmarks._timing import bench, format_row, print_header
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
    )


def _raw_attention_step(np_module, cfg: ReferenceConfig):
    """Batched QK^T, masked softmax, @V — no autograd, no Python Tensor nodes."""
    bh = cfg.batch * cfg.heads
    seq, dk = cfg.seq, cfg.d_k
    scale = dk ** -0.5
    q = np_module.random.standard_normal((bh, seq, dk)).astype(np_module.float32)
    k = np_module.random.standard_normal((bh, seq, dk)).astype(np_module.float32)
    v = np_module.random.standard_normal((bh, seq, dk)).astype(np_module.float32)
    k_t = np_module.ascontiguousarray(np_module.swapaxes(k, -2, -1))

    def step() -> None:
        scores = np_module.matmul(q, k_t) * scale
        mask = np_module.triu(np_module.full((seq, seq), -np_module.inf, dtype=scores.dtype), k=1)
        scores = scores + mask
        scores = scores - np_module.max(scores, axis=-1, keepdims=True)
        probs = np_module.exp(scores)
        probs = probs / np_module.sum(probs, axis=-1, keepdims=True)
        _ = np_module.matmul(probs, v)

    return step


def run_suite(
    cfg: ReferenceConfig,
    *,
    fused_blocks: bool = True,
    fused_trunk: bool = False,
) -> tuple[list[dict], dict, dict | None]:
    from NimbleML.neural_network.attention import MultiHeadAttention
    from NimbleML.utils.autograd_profile import profile_gpt_forward, profile_gpt_train_step
    from NimbleML.utils.mask import causal_mask_tensor
    from NimbleML.utils.np_backend import np, set_dtype, using_gpu
    from NimbleML.utils.tensor import Tensor

    set_dtype("float32")

    model = make_model(cfg, fused_blocks=fused_blocks, fused_trunk=fused_trunk)
    inputs, targets, tokens = make_inputs(cfg)
    warmup, runs = cfg.warmup, cfg.runs

    pos = model._absolute_pos_encoding(cfg.seq)

    def gpt_embed():
        _ = model.token_emb(inputs) + pos

    def gpt_blocks():
        x = model.token_emb(inputs) + pos
        _ = model.blocks(x)

    def gpt_forward():
        _ = model.forward(inputs)

    def gpt_forward_backward():
        zero_grads(model.parameters())
        loss = model.compute_loss(inputs, targets)
        loss.backward()
        model.clear_pos_encoding_cache()

    mha = MultiHeadAttention(cfg.d_model, cfg.heads)
    x = Tensor(
        np.random.standard_normal((cfg.batch, cfg.seq, cfg.d_model)).astype(np.float32).ravel(),
        (cfg.batch, cfg.seq, cfg.d_model),
        requires_grad=True,
    )
    mask = causal_mask_tensor(cfg.seq)

    def mha_fwd_bwd():
        zero_grads([x, *mha.parameters()])
        out = mha.forward(x, mask=mask)
        out.sum().backward()

    results = [
        bench(
            "raw_attention",
            _raw_attention_step(np, cfg),
            np_module=np,
            using_gpu=using_gpu,
            warmup=warmup,
            runs=runs,
        ),
        bench("mha_fwd_bwd", mha_fwd_bwd, np_module=np, using_gpu=using_gpu, warmup=warmup, runs=runs),
        bench("gpt_embed", gpt_embed, np_module=np, using_gpu=using_gpu, warmup=warmup, runs=runs, tokens=tokens),
        bench("gpt_blocks", gpt_blocks, np_module=np, using_gpu=using_gpu, warmup=warmup, runs=runs, tokens=tokens),
        bench("gpt_forward", gpt_forward, np_module=np, using_gpu=using_gpu, warmup=warmup, runs=runs, tokens=tokens),
        bench(
            "gpt_forward_backward",
            gpt_forward_backward,
            np_module=np,
            using_gpu=using_gpu,
            warmup=warmup,
            runs=runs,
            tokens=tokens,
        ),
    ]

    graph = profile_gpt_forward(model, inputs)
    train_graph = profile_gpt_train_step(model, inputs, targets)
    results.append(
        {
            "name": "autograd_nodes_forward",
            "mean_ms": float(graph["nodes"]),
            "p50_ms": float(graph["nodes"]),
            "p95_ms": float(graph["nodes"]),
            "note": "count, not milliseconds",
        }
    )
    results.append(
        {
            "name": "autograd_nodes_train",
            "mean_ms": float(train_graph["nodes"]),
            "p50_ms": float(train_graph["nodes"]),
            "p95_ms": float(train_graph["nodes"]),
            "note": "count, not milliseconds",
        }
    )
    return results, graph, train_graph


def _print_overhead_analysis(results: list[dict], graph: dict, train_graph: dict | None) -> None:
    by_name = {r["name"]: r for r in results}
    raw_ms = by_name["raw_attention"]["mean_ms"]
    mha_ms = by_name["mha_fwd_bwd"]["mean_ms"]
    gpt_fwd = by_name["gpt_forward"]["mean_ms"]
    gpt_fwb = by_name["gpt_forward_backward"]["mean_ms"]

    print("-" * 96)
    print("Overhead analysis (mean ms):")
    print(f"  MHA fwd+bwd vs raw attention matmul path: {mha_ms / raw_ms:.1f}x")
    if "tokens_per_sec" in by_name["gpt_forward"]:
        print(f"  GPT forward: {by_name['gpt_forward']['tokens_per_sec']:,.0f} tok/s")
    print(f"  GPT forward vs forward+backward: {gpt_fwb / gpt_fwd:.2f}x slower")
    print(f"  Autograd nodes on GPT forward: {int(graph['nodes'])}")
    if train_graph is not None:
        print(f"  Autograd nodes on GPT train step: {int(train_graph['nodes'])}")
        print(f"  Train-step within budget: {train_graph.get('within_budget')}")
    top_ops = list(graph["ops"].items())[:6]
    print("  Top forward ops:")
    for op, count in top_ops:
        print(f"    {op:30} {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward-only GPT / attention benchmarks.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="Print detailed train-step graph profile.")
    parser.add_argument("--no-fused-blocks", action="store_true", help="Use unfused TransformerBlock stack.")
    parser.add_argument("--fused-trunk", action="store_true", help="Fuse all blocks + final LN into one node.")
    args = parser.parse_args()
    cfg = _resolve_config(args.quick, args.warmup, args.runs)
    fused_blocks = not args.no_fused_blocks
    fused_trunk = args.fused_trunk

    from NimbleML.utils.autograd_profile import format_profile_report
    from NimbleML.utils.np_backend import dtype, using_gpu

    results, graph, train_graph = run_suite(cfg, fused_blocks=fused_blocks, fused_trunk=fused_trunk)
    dev_label = "GPU" if using_gpu else "CPU"
    print_header("Forward-only / autograd overhead", cfg, device=dev_label, dtype=str(dtype))
    for row in results:
        if row["name"].startswith("autograd_nodes"):
            print(f"{row['name']:28} {int(row['mean_ms']):>10} nodes")
        else:
            print(format_row(row))
    _print_overhead_analysis(results, graph, train_graph)
    if args.profile and train_graph is not None:
        print("-" * 96)
        print(format_profile_report(train_graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
