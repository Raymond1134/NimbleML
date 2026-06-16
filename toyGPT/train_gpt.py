"""Train a toy GPT on WikiText with config-driven hyperparameters."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig
from toyGPT.data import load_wikitext_splits, random_batch


def _configure_backend(cfg: ToyGPTConfig) -> None:
    os.environ["NIMBLEML_DEVICE"] = cfg.device
    os.environ["NIMBLEML_DTYPE"] = cfg.dtype


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as host_np

    host_np.random.seed(seed)


def _build_scheduler(optimizer, cfg: ToyGPTConfig):
    from NimbleML.optimizers import CosineAnnealing, LinearWarmup

    cosine_steps = max(1, cfg.max_steps - cfg.warmup_steps)
    eta_min = cfg.lr * 0.1
    inner = CosineAnnealing(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return LinearWarmup(inner, warmup_steps=cfg.warmup_steps)


def _vlog(cfg: ToyGPTConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg)


def _load_or_train_tokenizer(cfg: ToyGPTConfig, train_text: str, *, resume_dir: Path | None) -> tuple:
    from NimbleML.data.tokenizer import BPETokenizer

    if resume_dir is not None:
        tokenizer_path = resume_dir / "tokenizer.json"
        if tokenizer_path.is_file():
            _vlog(cfg, f"[data] loading tokenizer from checkpoint {tokenizer_path}")
            return BPETokenizer.load(tokenizer_path), False

    if cfg.tokenizer_path.is_file():
        _vlog(cfg, f"[data] loading tokenizer from {cfg.tokenizer_path}")
        return BPETokenizer.load(cfg.tokenizer_path), False

    max_chars = cfg.tokenizer_max_chars if cfg.tokenizer_max_chars > 0 else None
    _vlog(
        cfg,
        f"[data] training BPE | vocab_size={cfg.vocab_size} "
        f"max_train_chars={max_chars or 'all'}",
    )
    t0 = time.perf_counter()
    tokenizer = BPETokenizer()
    tokenizer.train(
        train_text,
        vocab_size=cfg.vocab_size,
        verbose=bool(cfg.verbose),
        max_train_chars=max_chars,
        log_every=cfg.bpe_log_every,
    )
    _vlog(cfg, f"[data] BPE train finished in {time.perf_counter() - t0:.2f}s")

    cfg.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.tokenizer_path)
    print(f"[data] saved tokenizer -> {cfg.tokenizer_path}")
    return tokenizer, True


def _eval_loss(model, loss_fn, val_ids, cfg: ToyGPTConfig, rng) -> float:
    total = 0.0
    for _ in range(cfg.eval_batches):
        inputs, targets = random_batch(
            val_ids, batch_size=cfg.batch_size, seq_len=cfg.seq_len, rng=rng
        )
        logits = model.forward(inputs)
        loss = loss_fn(logits, targets)
        total += float(loss.data[0])
    return total / cfg.eval_batches


def _sample_text(model, tokenizer, cfg: ToyGPTConfig, prompt_ids: list[int]) -> str:
    from NimbleML.utils.np_backend import np
    from NimbleML.utils.tensor import Tensor

    ids = list(prompt_ids)
    for _ in range(cfg.sample_chars):
        window = ids[-cfg.seq_len :]
        if not window:
            break
        if len(window) < cfg.seq_len:
            window = [0] * (cfg.seq_len - len(window)) + window
        inputs = Tensor(window, (1, cfg.seq_len))
        logits = model.forward(inputs)
        logits_arr = np.asarray(logits.data, dtype=np.float32).reshape(cfg.seq_len, -1)
        last = logits_arr[-1]
        if cfg.temperature <= 0:
            next_id = int(np.argmax(last))
        else:
            scaled = last / cfg.temperature
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= np.sum(probs)
            next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)
    return tokenizer.decode(ids)


def train(cfg: ToyGPTConfig, *, resume: str | None) -> None:
    _configure_backend(cfg)
    _seed_everything(cfg.seed)

    from NimbleML.losses import CrossEntropyLoss
    from NimbleML.models import GPT
    from NimbleML.optimizers import AdamW
    from NimbleML.utils.clip_grad import clip_grad_norm_
    from NimbleML.utils.np_backend import np, set_dtype

    set_dtype(cfg.dtype)
    rng = np.random.default_rng(cfg.seed)

    _vlog(cfg, f"[init] device={cfg.device} dtype={cfg.dtype} seed={cfg.seed}")
    _vlog(cfg, f"[init] config={cfg.config_path}")

    t0 = time.perf_counter()
    print(f"[data] loading {cfg.dataset} ...")
    train_text, val_text = load_wikitext_splits(cfg.dataset, cfg.cache_dir, cfg.data_dir)
    _vlog(
        cfg,
        f"[data] loaded splits | train_chars={len(train_text):,} val_chars={len(val_text):,} "
        f"elapsed={time.perf_counter() - t0:.2f}s",
    )

    resume_dir = None
    if resume:
        from toyGPT.checkpoint import resolve_resume_path

        resume_dir = resolve_resume_path(cfg.checkpoint_dir, resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {resume_dir}")
        _vlog(cfg, f"[ckpt] resume path={resume_dir}")

    tokenizer, bpe_just_trained = _load_or_train_tokenizer(cfg, train_text, resume_dir=resume_dir)

    max_chars = cfg.tokenizer_max_chars if cfg.tokenizer_max_chars > 0 else None
    t_enc = time.perf_counter()
    cached_train = tokenizer.take_train_corpus_ids() if bpe_just_trained and max_chars is None else None
    if cached_train is not None:
        train_ids = cached_train
        _vlog(cfg, f"[data] reusing BPE train corpus ids | tokens={len(train_ids):,}")
    else:
        _vlog(cfg, f"[data] encoding train split ...")
        train_ids = tokenizer.encode(train_text, verbose=bool(cfg.verbose))
        _vlog(cfg, f"[data] train encoded | tokens={len(train_ids):,}")

    _vlog(cfg, f"[data] encoding val split ...")
    val_ids = tokenizer.encode(val_text, verbose=bool(cfg.verbose))
    _vlog(
        cfg,
        f"[data] corpus tokenized | train_tokens={len(train_ids):,} val_tokens={len(val_ids):,} "
        f"encode_elapsed={time.perf_counter() - t_enc:.2f}s",
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
        model.parameters(),
        learning_rate=cfg.lr,
        beta1=0.9,
        beta2=0.95,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)
    loss_fn = CrossEntropyLoss()
    _vlog(
        cfg,
        f"[model] ready | params_vocab={vocab_size} d_model={cfg.d_model} "
        f"layers={cfg.n_layer} heads={cfg.n_head} seq={cfg.seq_len} batch={cfg.batch_size}",
    )

    step = 0
    best_val_loss: float | None = None
    checkpoint_root = cfg.checkpoint_dir

    if resume_dir is not None:
        from toyGPT.checkpoint import load_checkpoint

        state, tokenizer = load_checkpoint(resume_dir, model=model, optimizer=optimizer)
        step = int(state["step"])
        best_val_loss = state.get("best_val_loss")
        train_ids = tokenizer.encode(train_text, verbose=bool(cfg.verbose))
        val_ids = tokenizer.encode(val_text, verbose=bool(cfg.verbose))
        for _ in range(step):
            scheduler.step()
        model.clear_pos_encoding_cache()
        print(f"[ckpt] resumed from {resume_dir} at step {step}")

    tokens_per_step = cfg.batch_size * cfg.seq_len
    step_times: deque[float] = deque(maxlen=cfg.rolling_avg)
    step_tokens: deque[float] = deque(maxlen=cfg.rolling_avg)

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
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        model.clear_pos_encoding_cache()
        scheduler.step()
        step += 1

        elapsed = time.perf_counter() - t0
        step_ms = elapsed * 1000.0
        tok_s = tokens_per_step / elapsed if elapsed > 0 else 0.0
        step_times.append(step_ms)
        step_tokens.append(tok_s)
        loss_val = float(loss.data[0])
        lr = optimizer.get_lr()[0]

        if step % cfg.log_every == 0 or step == 1:
            avg_ms = sum(step_times) / len(step_times)
            avg_tok_s = sum(step_tokens) / len(step_tokens)
            print(
                f"step={step:6d} loss={loss_val:.4f} lr={lr:.2e} "
                f"tok/s={tok_s:,.0f} step_ms={step_ms:.1f} "
                f"avg_tok/s={avg_tok_s:,.0f} avg_ms={avg_ms:.1f}"
            )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_loss = _eval_loss(model, loss_fn, val_ids, cfg, rng)
            ppl = math.exp(min(val_loss, 20.0))
            print(f"[eval] step={step:6d} val_loss={val_loss:.4f} ppl={ppl:.2f}")

            if cfg.verbose:
                prompt = train_ids[: cfg.seq_len]
                sample = _sample_text(model, tokenizer, cfg, prompt)
                preview = sample[: cfg.sample_chars].replace("\n", "\\n")
                print(f"[eval] sample: {preview}")

            from toyGPT.checkpoint import copy_checkpoint, save_checkpoint

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_root / "best",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    config=cfg.to_dict(),
                    tokenizer=tokenizer,
                )
                print(f"[ckpt] saved best (val_loss={val_loss:.4f})")

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
            )
            copy_checkpoint(step_dir, checkpoint_root / "latest")
            print(f"[ckpt] saved step_{step} -> latest")

    print(f"[train] finished at step {step}")


def main(argv: list[str] | None = None) -> int:
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
