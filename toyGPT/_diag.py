"""Diagnostic harness: full-depth training steps with synchronous CUDA.

Run:  python toyGPT\_diag.py [steps] [batch_size]

Sets CUDA_LAUNCH_BLOCKING=1 so any illegal-memory-access reports at the true
faulting kernel (CUDA errors are otherwise async and surface at the next sync,
pointing at the wrong op). Logs loss + free/total VRAM each step and aborts on
NaN/Inf so we can separate "numeric blowup" from "memory corruption".
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Must be set before any CUDA context is created.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig  # noqa: E402
from toyGPT.data import random_batch  # noqa: E402
from toyGPT.fineweb import load_token_bin, prepare_corpus  # noqa: E402
from toyGPT.train_utils import adamw_param_groups, seed_everything  # noqa: E402

import numpy as host_np  # noqa: E402


def _mem_str():
    try:
        import cupy

        free, total = cupy.cuda.runtime.memGetInfo()
        used = total - free
        return f"vram_used={used/1e9:.2f}/{total/1e9:.2f}GB"
    except Exception:
        return "vram=?"


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = ToyGPTConfig.from_toml(TOYGPT_ROOT / "gpt_toy_config.toml")
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.batch_size
    cfg.batch_size = batch

    os.environ["NIMBLEML_DEVICE"] = cfg.device
    os.environ["NIMBLEML_DTYPE"] = cfg.dtype
    seed_everything(cfg.seed)

    from NimbleML.losses import CrossEntropyLoss
    from NimbleML.models import GPT
    from NimbleML.optimizers import AdamW
    from NimbleML.utils.clip_grad import clip_grad_norm_
    from NimbleML.utils.np_backend import apply_runtime_config, using_gpu

    apply_runtime_config(cfg.device, cfg.dtype)
    rng = host_np.random.default_rng(cfg.seed)

    print(f"[diag] device={cfg.device} gpu={using_gpu} batch={batch} steps={steps} "
          f"d_model={cfg.d_model} layers={cfg.n_layer} heads={cfg.n_head} seq={cfg.seq_len}")

    tokenizer, _tok_path, train_path, val_path, meta = prepare_corpus(cfg, verbose=False)
    train_ids = load_token_bin(train_path)
    vocab_size = tokenizer.vocab_size
    print(f"[diag] corpus train_tokens={meta['train_tokens']:,} vocab={vocab_size}")

    model = GPT(vocab_size, cfg.d_model, cfg.n_head, cfg.n_layer, cfg.seq_len, ff_mult=cfg.ff_mult)
    optimizer = AdamW(
        adamw_param_groups(model, lr=cfg.lr, weight_decay=cfg.weight_decay),
        learning_rate=cfg.lr, beta1=0.9, beta2=0.95, epsilon=1e-8, weight_decay=cfg.weight_decay,
    )
    loss_fn = CrossEntropyLoss()

    import gc
    import math

    from NimbleML.utils.np_backend import np as bnp

    def _finite(arr) -> bool:
        a = bnp.asarray(arr)
        return bool(bnp.isfinite(a).all())

    def _maxabs(arr) -> float:
        a = bnp.asarray(arr)
        return float(bnp.max(bnp.abs(a))) if a.size else 0.0

    # Flat parameter list -> human label, to locate the first bad gradient.
    labels = ["token_emb", "pos_emb"]
    for li in range(cfg.n_layer):
        labels += [f"blk{li}.ln1", f"blk{li}.q.w", f"blk{li}.q.b", f"blk{li}.k.w",
                   f"blk{li}.k.b", f"blk{li}.v.w", f"blk{li}.v.b", f"blk{li}.o.w",
                   f"blk{li}.o.b", f"blk{li}.ln2", f"blk{li}.ff1.w", f"blk{li}.ff1.b",
                   f"blk{li}.ff2.w", f"blk{li}.ff2.b"]
    labels += ["final_ln"]

    expected = math.log(vocab_size)
    print(f"[diag] expected step-1 loss ~= ln(vocab) = {expected:.3f}")

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = random_batch(train_ids, batch_size=cfg.batch_size, seq_len=cfg.seq_len, rng=rng)
        ids_min = int(bnp.asarray(inputs.data).min())
        ids_max = int(bnp.asarray(inputs.data).max())
        logits = model.forward(inputs)
        logit_max = _maxabs(logits.data)
        logits_finite = _finite(logits.data)
        loss = loss_fn(logits, targets)
        loss.backward()
        loss_val = float(loss.data[0])

        params = model.parameters()
        first_bad = None
        for idx, p in enumerate(params):
            if p.grad is not None and not _finite(p.grad):
                first_bad = idx
                break

        del logits, loss, inputs, targets
        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        model.clear_pos_encoding_cache()
        gc.collect()
        if using_gpu:
            import cupy

            cupy.cuda.Device().synchronize()

        gn = float(grad_norm) if grad_norm is not None else float("nan")
        bad_str = "" if first_bad is None else f" FIRST_BAD_GRAD=#{first_bad}:{labels[first_bad]}"
        fin_str = "" if logits_finite else " LOGITS_NONFINITE"
        print(f"step={step:3d} loss={loss_val:.4f} grad_norm={gn:.3f} "
              f"logit_maxabs={logit_max:.2f} ids=[{ids_min},{ids_max}] {_mem_str()}{fin_str}{bad_str}")

        if not math.isfinite(loss_val) or first_bad is not None or not logits_finite:
            print(f"[diag] FAIL: non-finite at step {step} "
                  f"(loss_finite={math.isfinite(loss_val)} logits_finite={logits_finite} "
                  f"first_bad_grad={first_bad})")
            return 2

    print("[diag] OK: completed all steps with finite loss and no CUDA fault.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("[diag] EXCEPTION (true location under CUDA_LAUNCH_BLOCKING):")
        traceback.print_exc()
        raise SystemExit(1)
