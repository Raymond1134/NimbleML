"""NimbleML performance benchmarks — GPU-first, GPT training focused.

Run:
  python benchmark.py                    # fast quick GPT smoke (~seconds)
  python benchmark.py --train-shape  # match toyGPT config (slow)
  python benchmark.py --full       # all micro-benchmarks (quick shapes)
  python benchmark.py --full --train-shape
  python benchmark.py --compare-torch
  python benchmark.py --count-graph
"""

from __future__ import annotations

import argparse
import json
import os

# Default to GPU before NimbleML backend initializes.
os.environ.setdefault("NIMBLEML_DEVICE", "gpu")

import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NimbleML.activations import Softmax
from NimbleML.layers import Embedding, LayerNorm, MaxPool2D
from NimbleML.layers.conv2d import Conv2D
from NimbleML.layers.dense import Dense
from NimbleML.losses import CrossEntropyLoss
from NimbleML.models.gpt import GPT
from NimbleML.neural_network.attention import Attention, causal_mask_tensor
from NimbleML.neural_network.feed_forward import FeedForward
from NimbleML.optimizers import Adam, AdamW, SGD, StepLR
from NimbleML.utils.autograd_profile import profile_gpt_forward
from NimbleML.utils.clip_grad import clip_grad_norm_
from NimbleML.utils.np_backend import np, set_dtype, using_gpu
from NimbleML.utils.saveload import save
from NimbleML.utils.tensor import Tensor

BenchFn = Callable[[], None]


@dataclass(frozen=True)
class GptBenchConfig:
    vocab: int
    d_model: int
    heads: int
    layers: int
    seq: int
    batch: int
    warmup: int
    runs: int

    @property
    def tokens_per_step(self) -> float:
        return float(self.batch * self.seq)


# Quick default — fast smoke benchmark (seconds, not minutes).
QUICK_GPT = GptBenchConfig(
    vocab=4096,
    d_model=512,
    heads=8,
    layers=8,
    seq=256,
    batch=4,
    warmup=3,
    runs=5,
)

# Matches toyGPT/gpt_toy_config.toml — use with --train-shape only.
TRAIN_GPT = GptBenchConfig(
    vocab=16384,
    d_model=512,
    heads=8,
    layers=18,
    seq=256,
    batch=16,
    warmup=2,
    runs=5,
)

WEIGHTS = {
    "gpt": {
        "gpt_fwd": 1.00,
        "gpt_fwd_bwd": 1.00,
        "gpt_train_step": 1.00,
    },
    "text": {
        "tensor_add_bwd": 0.40,
        "dense_fwd_bwd": 0.90,
        "conv2d_fwd_bwd": 0.05,
        "maxpool2d_fwd_bwd": 0.05,
        "embedding_lookup_fwd": 0.85,
        "layernorm_fwd_bwd": 0.80,
        "softmax_fwd": 0.75,
        "attention_fwd_seq128": 0.95,
        "attention_fwd_seq256": 1.00,
        "feedforward_fwd_bwd": 0.95,
        "cross_entropy_3d_fwd_bwd": 0.90,
        "optimizer_adam_step": 0.85,
        "scheduler_step_lr_step": 0.10,
        "gpt_fwd": 1.00,
        "gpt_fwd_bwd": 1.00,
        "gpt_train_step": 1.00,
        "checkpoint_save_dense": 0.10,
    },
}


def _sync_if_gpu() -> None:
    if using_gpu and hasattr(np, "cuda"):
        np.cuda.Stream.null.synchronize()


def _bench(
    name: str,
    fn: BenchFn,
    *,
    warmup: int = 2,
    runs: int = 5,
    tokens: float | None = None,
) -> dict:
    for _ in range(warmup):
        fn()
    _sync_if_gpu()

    timings_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync_if_gpu()
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


def _make_gpt_inputs(cfg: GptBenchConfig):
    batch, seq_len, vocab = cfg.batch, cfg.seq, cfg.vocab
    inputs = Tensor.from_int64(
        np.random.randint(0, vocab, size=(batch, seq_len), dtype=np.int64).ravel(),
        (batch, seq_len),
    )
    targets = Tensor.from_int64(
        np.random.randint(0, vocab, size=(batch, seq_len), dtype=np.int64).ravel(),
        (batch, seq_len),
    )
    return inputs, targets, float(batch * seq_len)


def _make_gpt_model(cfg: GptBenchConfig) -> GPT:
    return GPT(cfg.vocab, cfg.d_model, cfg.heads, cfg.layers, cfg.seq)


def _zero_param_grads(params, *, set_to_none: bool = True) -> None:
    for param in params:
        if set_to_none:
            param.grad = None
        else:
            param.zero_grad()


def run_gpt_suite(cfg: GptBenchConfig) -> list[dict]:
    """One shared model; forward, backward, and train-step timings."""
    model = _make_gpt_model(cfg)
    inputs, targets, tokens = _make_gpt_inputs(cfg)
    loss_fn = CrossEntropyLoss()
    opt = AdamW(model.parameters(), learning_rate=3e-4, weight_decay=0.1)
    warmup, runs = cfg.warmup, cfg.runs

    def forward_step():
        _ = model.forward(inputs)

    def forward_backward_step():
        _zero_param_grads(model.parameters())
        logits = model.forward(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()

    def train_step():
        opt.zero_grad(set_to_none=True)
        logits = model.forward(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.clear_pos_encoding_cache()

    return [
        _bench("gpt_fwd", forward_step, warmup=warmup, runs=runs, tokens=tokens),
        _bench("gpt_fwd_bwd", forward_backward_step, warmup=warmup, runs=runs, tokens=tokens),
        _bench("gpt_train_step", train_step, warmup=warmup, runs=runs, tokens=tokens),
    ]


def print_gpt_graph_profile(cfg: GptBenchConfig) -> None:
    """Count autograd nodes for one GPT forward (reduction target baseline)."""
    model = _make_gpt_model(cfg)
    inputs, _, _ = _make_gpt_inputs(cfg)
    stats = profile_gpt_forward(model, inputs)
    print(f"\nGPT forward autograd graph: {stats['nodes']} nodes")
    print(f"logits shape: {stats['logits_shape']}")
    print("op counts:")
    for op, count in stats["ops"].items():
        print(f"  {op:32} {count}")
    print("Target: reduce node count as Tier 3 fusion continues (attention/FFN/CE done).")


def _tensor_add_bwd_case() -> dict:
    a = Tensor(np.random.standard_normal((256, 256)).astype(np.float32).ravel(), (256, 256), requires_grad=True)
    b = Tensor(np.random.standard_normal((256, 256)).astype(np.float32).ravel(), (256, 256), requires_grad=True)

    def step():
        a.grad = None
        b.grad = None
        (a + b).sum().backward()

    return _bench("tensor_add_bwd", step)


def _dense_case() -> dict:
    layer = Dense(512, 512)
    x = Tensor(np.random.standard_normal((128, 512)).astype(np.float32).ravel(), (128, 512), requires_grad=True)

    def step():
        x.grad = None
        layer.weights.grad = None
        layer.biases.grad = None
        layer.forward(x).sum().backward()

    return _bench("dense_fwd_bwd", step)


def _conv2d_case() -> dict:
    layer = Conv2D(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
    x = Tensor(np.random.standard_normal((8, 32, 32, 32)).astype(np.float32).ravel(), (8, 32, 32, 32), requires_grad=True)

    def step():
        x.grad = None
        layer.weights.grad = None
        layer.biases.grad = None
        layer.forward(x).sum().backward()

    return _bench("conv2d_fwd_bwd", step)


def _maxpool2d_case() -> dict:
    layer = MaxPool2D(kernel_size=2, stride=2)
    x = Tensor(np.random.standard_normal((16, 32, 32, 32)).astype(np.float32).ravel(), (16, 32, 32, 32), requires_grad=True)

    def step():
        x.grad = None
        layer.forward(x).sum().backward()

    return _bench("maxpool2d_fwd_bwd", step)


def _embedding_case(cfg: GptBenchConfig) -> dict:
    layer = Embedding(vocab_size=cfg.vocab, embed_dim=cfg.d_model)
    ids = np.random.randint(0, cfg.vocab, size=(cfg.batch, cfg.seq)).tolist()

    def step():
        _ = layer.forward(ids)

    return _bench("embedding_lookup_fwd", step, tokens=cfg.tokens_per_step)


def _layernorm_case(cfg: GptBenchConfig) -> dict:
    ln = LayerNorm(cfg.d_model)
    x = Tensor(
        np.random.standard_normal((cfg.batch, cfg.seq, cfg.d_model)).astype(np.float32).ravel(),
        (cfg.batch, cfg.seq, cfg.d_model),
        requires_grad=True,
    )

    def step():
        x.grad = None
        ln.gamma.grad = None
        ln.beta.grad = None
        ln.forward(x).sum().backward()

    return _bench("layernorm_fwd_bwd", step, tokens=cfg.tokens_per_step)


def _softmax_case(cfg: GptBenchConfig) -> dict:
    sm = Softmax(axis=-1)
    logits = Tensor(
        np.random.standard_normal((cfg.batch, cfg.seq, cfg.vocab)).astype(np.float32).ravel(),
        (cfg.batch, cfg.seq, cfg.vocab),
    )

    def step():
        _ = sm(logits)

    return _bench("softmax_fwd", step, tokens=cfg.tokens_per_step)


def _attention_case(cfg: GptBenchConfig, seq_len: int) -> dict:
    batch, d_k = cfg.batch, cfg.d_model // cfg.heads
    q = Tensor(np.random.standard_normal((batch, seq_len, d_k)).astype(np.float32).ravel(), (batch, seq_len, d_k))
    k = Tensor(np.random.standard_normal((batch, seq_len, d_k)).astype(np.float32).ravel(), (batch, seq_len, d_k))
    v = Tensor(np.random.standard_normal((batch, seq_len, d_k)).astype(np.float32).ravel(), (batch, seq_len, d_k))
    attn = Attention(d_k)

    def step():
        _ = attn.forward(q, k, v, mask=causal_mask_tensor(seq_len))

    return _bench(f"attention_fwd_seq{seq_len}", step, tokens=float(batch * seq_len))


def _feedforward_case(cfg: GptBenchConfig) -> dict:
    ff = FeedForward(cfg.d_model, ff_mult=4)
    x = Tensor(
        np.random.standard_normal((cfg.batch, cfg.seq, cfg.d_model)).astype(np.float32).ravel(),
        (cfg.batch, cfg.seq, cfg.d_model),
        requires_grad=True,
    )

    def step():
        x.grad = None
        for p in ff.parameters():
            p.grad = None
        ff.forward(x).sum().backward()

    return _bench("feedforward_fwd_bwd", step, tokens=cfg.tokens_per_step)


def _cross_entropy_case(cfg: GptBenchConfig) -> dict:
    loss_fn = CrossEntropyLoss()
    logits = Tensor(
        np.random.standard_normal((cfg.batch, cfg.seq, cfg.vocab)).astype(np.float32).ravel(),
        (cfg.batch, cfg.seq, cfg.vocab),
        requires_grad=True,
    )
    targets = Tensor(
        np.random.randint(0, cfg.vocab, size=(cfg.batch, cfg.seq), dtype=np.int64).ravel(),
        (cfg.batch, cfg.seq),
    )

    def step():
        logits.grad = None
        loss_fn(logits, targets).backward()

    return _bench(
        "cross_entropy_3d_fwd_bwd",
        step,
        warmup=min(2, cfg.warmup),
        runs=min(5, cfg.runs),
        tokens=cfg.tokens_per_step,
    )


def _optimizer_case(cfg: GptBenchConfig) -> dict:
    model = _make_gpt_model(cfg)
    opt = Adam(model.parameters(), learning_rate=3e-4)

    def step():
        for p in model.parameters():
            p.grad = np.random.standard_normal(p.size).astype(np.float32)
        opt.step()

    return _bench("optimizer_adam_step", step, warmup=1, runs=min(5, cfg.runs))


def _scheduler_case() -> dict:
    opt = SGD([Tensor([1.0], (1,), requires_grad=True)], learning_rate=1.0)
    sched = StepLR(opt, step_size=20, gamma=0.5)

    def step():
        sched.step()

    return _bench("scheduler_step_lr_step", step, warmup=2, runs=10)


def _checkpoint_save_case() -> dict:
    model = Dense(1024, 1024)
    tmp_path = ROOT / "tests" / "_bench_ckpt_dense.npz"

    def step():
        save(model, tmp_path)
        if tmp_path.exists():
            tmp_path.unlink()

    return _bench("checkpoint_save_dense", step, warmup=2, runs=6)


def run_full_suite(cfg: GptBenchConfig) -> list[dict]:
    return [
        _tensor_add_bwd_case(),
        _dense_case(),
        _conv2d_case(),
        _maxpool2d_case(),
        _embedding_case(cfg),
        _layernorm_case(cfg),
        _softmax_case(cfg),
        _attention_case(cfg, 128),
        _attention_case(cfg, min(256, cfg.seq)),
        _feedforward_case(cfg),
        _cross_entropy_case(cfg),
        _optimizer_case(cfg),
        _scheduler_case(),
        *run_gpt_suite(cfg),
        _checkpoint_save_case(),
    ]


def _rank(results: list[dict], profile: str) -> list[dict]:
    weights = WEIGHTS.get(profile, WEIGHTS["text"])
    ranked = []
    for r in results:
        weight = weights.get(r["name"], 0.3)
        ranked.append({**r, "weight": weight, "priority_score": r["mean_ms"] * weight})
    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    return ranked


def print_report(results: list[dict], profile: str, cfg: GptBenchConfig) -> None:
    device = "GPU" if using_gpu else "CPU"
    dtype = os.environ.get("NIMBLEML_DTYPE", "float32")
    ranked = _rank(results, profile)
    print(f"\nNimbleML benchmark ({device}, dtype={dtype})")
    print(
        f"GPT config: vocab={cfg.vocab} d_model={cfg.d_model} layers={cfg.layers} "
        f"heads={cfg.heads} batch={cfg.batch} seq={cfg.seq} "
        f"(warmup={cfg.warmup} runs={cfg.runs})"
    )
    print(f"Profile: {profile}")
    print("-" * 120)
    print(f"{'op':28} {'mean ms':>10} {'p50 ms':>10} {'p95 ms':>10} {'tok/s':>12} {'priority':>10}")
    print("-" * 120)
    weights = WEIGHTS.get(profile, WEIGHTS["text"])
    for r in results:
        tps = f"{r['tokens_per_sec']:,.0f}" if "tokens_per_sec" in r else "-"
        priority = r["mean_ms"] * weights.get(r["name"], 0.3)
        print(f"{r['name']:28} {r['mean_ms']:10.3f} {r['p50_ms']:10.3f} {r['p95_ms']:10.3f} {tps:>12} {priority:10.3f}")
    print("-" * 120)
    print("Top optimization targets:")
    for idx, r in enumerate(ranked[:5], start=1):
        tps = f", {r['tokens_per_sec']:,.0f} tok/s" if "tokens_per_sec" in r else ""
        print(f"  {idx}) {r['name']} — {r['mean_ms']:.3f} ms{tps}")


def compare_torch_gpt(cfg: GptBenchConfig) -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed; skip --compare-torch (pip install torch).")
        return

    if not torch.cuda.is_available():
        print("PyTorch CUDA unavailable; CPU comparison only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, seq = cfg.batch, cfg.seq
    vocab, d_model, n_layer, n_head = cfg.vocab, cfg.d_model, cfg.layers, cfg.heads
    warmup, runs = cfg.warmup, cfg.runs

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab, d_model)
            self.pos_emb = nn.Embedding(seq, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=4 * d_model, batch_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.lm_head = nn.Linear(d_model, vocab)

        def forward(self, x):
            pos = torch.arange(seq, device=x.device)
            return self.lm_head(self.blocks(self.tok_emb(x) + self.pos_emb(pos)))

    model = TinyGPT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)
    tokens = float(batch * seq)

    def train_step():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # warmup
    for _ in range(warmup):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        train_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_ms = statistics.mean(times) * 1000.0
    tps = tokens / (mean_ms / 1000.0)
    print(f"\nPyTorch reference train step: {mean_ms:.3f} ms, {tps:,.0f} tok/s ({device})")
    print("Goal: match or beat PyTorch tok/s on gpt_train_step after Tier 3 optimizations.")


def _resolve_gpt_config(
    *,
    train_shape: bool,
    warmup: Optional[int],
    runs: Optional[int],
) -> GptBenchConfig:
    base = TRAIN_GPT if train_shape else QUICK_GPT
    if warmup is None and runs is None:
        return base
    return GptBenchConfig(
        vocab=base.vocab,
        d_model=base.d_model,
        heads=base.heads,
        layers=base.layers,
        seq=base.seq,
        batch=base.batch,
        warmup=base.warmup if warmup is None else warmup,
        runs=base.runs if runs is None else runs,
    )


def run_once(*, full: bool, cfg: GptBenchConfig) -> list[dict]:
    set_dtype("float32")
    return run_full_suite(cfg) if full else run_gpt_suite(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="NimbleML benchmarks (quick GPT smoke by default).")
    parser.add_argument("--full", action="store_true", help="Run full micro-benchmark suite.")
    parser.add_argument(
        "--train-shape",
        action="store_true",
        help="Use toyGPT training dimensions (18L/512d/16k vocab — slow).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend.")
    parser.add_argument("--profile", choices=("gpt", "text"), default="gpt")
    parser.add_argument("--count-graph", action="store_true", help="Print GPT forward autograd node counts.")
    parser.add_argument("--compare-torch", action="store_true", help="Compare train-step tok/s vs PyTorch.")
    parser.add_argument("--json", type=str, default="", help="Write results JSON to path.")
    parser.add_argument("--both", action="store_true", help="Run CPU and GPU subprocesses.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup iterations.")
    parser.add_argument("--runs", type=int, default=None, help="Override timed run count.")
    args = parser.parse_args()
    cfg = _resolve_gpt_config(train_shape=args.train_shape, warmup=args.warmup, runs=args.runs)

    if args.both:
        base = os.environ.copy()
        for dev in ("cpu", "gpu"):
            env = dict(base)
            env["NIMBLEML_DEVICE"] = dev
            cmd = [sys.executable, __file__]
            if args.full:
                cmd.append("--full")
            if args.train_shape:
                cmd.append("--train-shape")
            cmd.extend(["--profile", args.profile])
            print(f"\n=== {dev.upper()} ===")
            subprocess.run(cmd, env=env, check=False)
        return 0

    if args.cpu:
        os.environ["NIMBLEML_DEVICE"] = "cpu"
        import importlib
        import NimbleML.utils.np_backend as nb
        importlib.reload(nb)

    graph_only = (
        args.count_graph
        and not args.full
        and not args.both
        and not args.json
        and not args.compare_torch
    )
    if graph_only:
        set_dtype("float32")
        print_gpt_graph_profile(cfg)
        return 0

    results = run_once(full=args.full, cfg=cfg)
    print_report(results, args.profile, cfg)
    if args.count_graph:
        print_gpt_graph_profile(cfg)
    if args.compare_torch:
        compare_torch_gpt(cfg)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
