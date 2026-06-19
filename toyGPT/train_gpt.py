"""Train a toy GPT on WikiText with config-driven hyperparameters."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as host_np  # host RNG for batch sampling (state is checkpointable)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig
from toyGPT.data import random_batch
from toyGPT.fineweb import load_token_bin, prepare_corpus
from toyGPT.sampling import prompt_ids_from_corpus, sample_text
from toyGPT.loss_history import LossHistory
from toyGPT.train_utils import adamw_param_groups, load_rng_state, save_rng_state, seed_everything


def _configure_backend(cfg: ToyGPTConfig) -> None:
    os.environ["NIMBLEML_DEVICE"] = cfg.device
    os.environ["NIMBLEML_DTYPE"] = cfg.dtype


def _build_scheduler(optimizer, cfg: ToyGPTConfig):
    from NimbleML.optimizers import CosineAnnealing, LinearWarmup

    cosine_steps = max(1, cfg.max_steps - cfg.warmup_steps)
    eta_min = 0.0
    inner = CosineAnnealing(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return LinearWarmup(inner, warmup_steps=cfg.warmup_steps), eta_min, cosine_steps


def _vlog(cfg: ToyGPTConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg)


def _eval_loss(model, loss_fn, val_ids, cfg: ToyGPTConfig, rng) -> float:
    from NimbleML.utils.np_backend import using_gpu

    total = 0.0
    for _ in range(cfg.eval_batches):
        inputs, targets = random_batch(
            val_ids, batch_size=cfg.batch_size, seq_len=cfg.seq_len, rng=rng
        )
        logits = model.forward(inputs)
        loss = loss_fn(logits, targets)
        total += float(loss.data[0])
        # Forward-only graphs still build backward closures (reference cycles);
        # release each one before the next eval batch to bound GPU memory.
        del logits, loss, inputs, targets
    gc.collect()
    if using_gpu:
        import cupy

        cupy.cuda.Device().synchronize()
    return total / cfg.eval_batches


def _cleanup_gpu(using_gpu: bool) -> None:
    gc.collect()
    if using_gpu:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.Device().synchronize()


def train(cfg: ToyGPTConfig, *, resume: str | None) -> None:
    _configure_backend(cfg)
    seed_everything(cfg.seed)

    from NimbleML.losses import CrossEntropyLoss
    from NimbleML.models import GPT
    from NimbleML.optimizers import AdamW
    from NimbleML.utils.clip_grad import clip_grad_norm_
    from NimbleML.utils.np_backend import apply_runtime_config, using_gpu

    apply_runtime_config(cfg.device, cfg.dtype)
    # Host RNG: batch sampling happens on the CPU over a memmapped corpus, and a
    # host NumPy generator's state serializes cleanly into checkpoints.
    rng = host_np.random.default_rng(cfg.seed)

    _vlog(cfg, f"[init] device={cfg.device} dtype={cfg.dtype} backend={'gpu' if using_gpu else 'cpu'} seed={cfg.seed}")
    if cfg.dtype != "float32":
        print(f"[init] warning: training dtype is {cfg.dtype!r}; float32 is recommended on GPU.")

    _vlog(cfg, f"[init] config={cfg.config_path}")

    resume_dir = None
    if resume:
        from toyGPT.checkpoint import resolve_resume_path

        resume_dir = resolve_resume_path(cfg.checkpoint_dir, resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {resume_dir}")
        _vlog(cfg, f"[ckpt] resume path={resume_dir}")

    t0 = time.perf_counter()
    print(f"[data] preparing {cfg.dataset} ({cfg.hf_subset}) ...")
    tokenizer, tokenizer_path, train_path, val_path, meta = prepare_corpus(
        cfg, resume_dir=resume_dir, verbose=bool(cfg.verbose)
    )
    train_ids = load_token_bin(train_path)
    val_ids = load_token_bin(val_path)
    _vlog(
        cfg,
        f"[data] corpus ready | train_tokens={meta['train_tokens']:,} "
        f"val_tokens={meta['val_tokens']:,} elapsed={time.perf_counter() - t0:.1f}s",
    )
    _vlog(
        cfg,
        f"[data] corpus memmapped on host (uint16); batches move to "
        f"{'GPU' if using_gpu else 'CPU'} as int64",
    )

    vocab_size = tokenizer.vocab_size

    if cfg.vocab_size != vocab_size:
        print(f"[model] note: config vocab_size={cfg.vocab_size}, tokenizer has {vocab_size}")

    _vlog(cfg, "[model] building GPT ...")
    model = GPT(
        vocab_size,
        cfg.d_model,
        cfg.n_head,
        cfg.n_layer,
        cfg.seq_len,
        ff_mult=cfg.ff_mult,
    )
    optimizer = AdamW(
        adamw_param_groups(model, lr=cfg.lr, weight_decay=cfg.weight_decay),
        learning_rate=cfg.lr,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=cfg.weight_decay,
    )
    scheduler, eta_min, cosine_steps = _build_scheduler(optimizer, cfg)
    loss_fn = CrossEntropyLoss()
    _vlog(
        cfg,
        f"[model] ready | params_vocab={vocab_size} d_model={cfg.d_model} "
        f"layers={cfg.n_layer} heads={cfg.n_head} seq={cfg.seq_len} batch={cfg.batch_size}",
    )
    _vlog(
        cfg,
        f"[train] LR schedule: warmup={cfg.warmup_steps} steps, then cosine "
        f"T_max={cosine_steps} lr={cfg.lr:.2e} -> eta_min={eta_min:.2e}",
    )

    step = 0
    best_val_loss: float | None = None
    evals_without_improvement = 0
    checkpoint_root = cfg.checkpoint_dir
    loss_history = LossHistory(checkpoint_root / "loss_history.csv")

    if resume_dir is not None:
        from toyGPT.checkpoint import load_checkpoint

        state, tokenizer = load_checkpoint(resume_dir, model=model, optimizer=optimizer)
        step = int(state["step"])
        best_val_loss = state.get("best_val_loss")
        rng_path = resume_dir / "rng.json"
        if rng_path.is_file():
            load_rng_state(rng_path, rng)
        # Fast-forward scheduler without O(step) Python loop on long resumes.
        scheduler.last_epoch = step - 1
        optimizer.set_lr(scheduler.get_lr())
        model.clear_pos_encoding_cache()
        loss_history.truncate_after(step)
        print(f"[ckpt] resumed from {resume_dir} at step {step}")

    tokens_per_step = cfg.batch_size * cfg.seq_len
    step_times: deque[float] = deque(maxlen=cfg.rolling_avg)
    step_tokens: deque[float] = deque(maxlen=cfg.rolling_avg)
    unstable_steps = 0

    print(
        f"[train] start | max_steps={cfg.max_steps} tokens/step={tokens_per_step} "
        f"log_every={cfg.log_every} eval_every={cfg.eval_every} ckpt_every={cfg.checkpoint_every}"
    )

    while step < cfg.max_steps:
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        inputs, targets = random_batch(
            train_ids, batch_size=cfg.batch_size, seq_len=cfg.seq_len, rng=rng
        )
        logits = model.forward(inputs)
        loss = loss_fn(logits, targets)
        try:
            loss.backward()
        except Exception as exc:
            print(f"[train] backward failed at step {step + 1}: {exc}")
            optimizer.zero_grad(set_to_none=True)
            model.clear_pos_encoding_cache()
            _cleanup_gpu(using_gpu)
            unstable_steps += 1
            if unstable_steps >= 3:
                print(
                    "[train] aborting: CUDA context likely poisoned. "
                    "Close this terminal, open a fresh one, and resume from your last good checkpoint."
                )
                raise
            step += 1
            continue

        loss_val = float(loss.data[0])
        del logits, loss, inputs, targets

        if not math.isfinite(loss_val):
            print(f"[train] warning: non-finite loss={loss_val} at step {step + 1}, skipping update")
            optimizer.zero_grad(set_to_none=True)
            model.clear_pos_encoding_cache()
            _cleanup_gpu(using_gpu)
            unstable_steps += 1
            if unstable_steps >= 5:
                print(
                    "[train] aborting after repeated non-finite loss. "
                    "Resume from an earlier checkpoint (weights may be corrupted)."
                )
                raise RuntimeError("training unstable: non-finite loss")
            step += 1
            continue

        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not math.isfinite(grad_norm):
            print(f"[train] warning: non-finite grad_norm at step {step + 1}, skipping update")
            optimizer.zero_grad(set_to_none=True)
            model.clear_pos_encoding_cache()
            _cleanup_gpu(using_gpu)
            unstable_steps += 1
            if unstable_steps >= 5:
                print(
                    "[train] aborting after repeated non-finite gradients. "
                    "Resume from an earlier checkpoint (e.g. step_10000 or best)."
                )
                raise RuntimeError("training unstable: non-finite grad_norm")
            step += 1
            continue

        unstable_steps = 0
        model.clear_pos_encoding_cache()
        optimizer.step()
        scheduler.step()
        if cfg.gc_every > 0 and step % cfg.gc_every == 0:
            gc.collect()
        if using_gpu:
            import cupy

            # Sync when logging (accurate tok/s) or after GC (reclaim closure cycles).
            if step % cfg.log_every == 0 or (cfg.gc_every > 0 and step % cfg.gc_every == 0):
                cupy.cuda.Device().synchronize()
        step += 1
        loss_history.maybe_record(step, loss_val)

        elapsed = time.perf_counter() - t0
        step_ms = elapsed * 1000.0
        tok_s = tokens_per_step / elapsed if elapsed > 0 else 0.0
        step_times.append(step_ms)
        step_tokens.append(tok_s)
        lr = optimizer.get_lr()[0]

        if step == cfg.warmup_steps:
            print(
                f"[train] warmup complete at step {step}; cosine decay active "
                f"(lr={lr:.2e}, target eta_min={eta_min:.2e} at step {cfg.max_steps})"
            )

        log_grad = cfg.log_grad_norm and (
            cfg.log_grad_norm_until_step <= 0 or step <= cfg.log_grad_norm_until_step
        )
        if step % cfg.log_every == 0 or step == 1:
            avg_ms = sum(step_times) / len(step_times)
            avg_tok_s = sum(step_tokens) / len(step_tokens)
            grad_msg = f" grad_norm={grad_norm:.3f}" if log_grad else ""
            print(
                f"step={step:6d} loss={loss_val:.4f} lr={lr:.2e}{grad_msg} "
                f"tok/s={tok_s:,.0f} step_ms={step_ms:.1f} "
                f"avg_tok/s={avg_tok_s:,.0f} avg_ms={avg_ms:.1f}"
            )
            if loss_val > 100.0 and step > cfg.warmup_steps:
                print("[train] warning: loss spike — consider lowering lr or increasing warmup.")

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_loss = _eval_loss(model, loss_fn, val_ids, cfg, rng)
            ppl = math.exp(min(val_loss, 20.0))
            print(f"[eval] step={step:6d} val_loss={val_loss:.4f} ppl={ppl:.2f}")

            if cfg.verbose:
                prompt = prompt_ids_from_corpus(val_ids, cfg.seq_len, rng=rng)
                sample_rng = host_np.random.default_rng(cfg.seed + step)
                preview = sample_text(
                    model,
                    tokenizer,
                    prompt_ids=prompt,
                    seq_len=cfg.seq_len,
                    max_new_tokens=cfg.sample_chars,
                    temperature=cfg.temperature,
                    rng=sample_rng,
                    include_prompt=False,
                )[: cfg.sample_chars].replace("\n", "\\n")
                print(f"[eval] sample: {preview}")

            from toyGPT.checkpoint import copy_checkpoint, save_checkpoint

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                evals_without_improvement = 0
                save_checkpoint(
                    checkpoint_root / "best",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    config=cfg.to_dict(),
                    tokenizer=tokenizer,
                    rng=rng,
                )
                print(f"[ckpt] saved best (val_loss={val_loss:.4f})")
            else:
                evals_without_improvement += 1
                if cfg.early_stop_patience > 0 and evals_without_improvement >= cfg.early_stop_patience:
                    print(
                        f"[train] early stop: val_loss flat for {evals_without_improvement} evals "
                        f"(patience={cfg.early_stop_patience})"
                    )
                    break

        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            from toyGPT.checkpoint import copy_checkpoint, save_checkpoint

            step_dir = checkpoint_root / f"step_{step}"
            save_checkpoint(
                step_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                config=cfg.to_dict(),
                tokenizer=tokenizer,
                rng=rng,
            )
            copy_checkpoint(step_dir, checkpoint_root / "latest")
            print(f"[ckpt] saved step_{step} -> latest")

    print(f"[train] finished at step {step}")


def main(argv: list[str] | None = None) -> int:
    # Generated samples can contain characters outside the Windows console's
    # default cp1252 codepage; print them with replacement instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Train a toy GPT on WikiText with BPE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=TOYGPT_ROOT / "gpt_toy_config.toml",
        help="Path to training config TOML.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from checkpoint dir, or 'latest' / 'best'.",
    )
    args = parser.parse_args(argv)

    cfg = ToyGPTConfig.from_toml(args.config.resolve())
    resume = args.resume.strip() or None
    train(cfg, resume=resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
