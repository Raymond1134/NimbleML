"""Checkpoint save/load for multi-session toy GPT training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as host_np

from typing import Union

from NimbleML.optimizers import Adam, AdamW
from NimbleML.utils import np_backend
from NimbleML.utils.saveload import load, save
from toyGPT.fast_tokenizer import FastBPETokenizer

OptimizerType = Union[Adam, AdamW]


def save_optimizer(optimizer: OptimizerType, path: Path) -> None:
    def to_host_array(arr) -> host_np.ndarray:
        get = getattr(arr, "get", None)
        if get is not None:
            arr = get()
        return host_np.asarray(arr)

    state: dict[str, Any] = {
        "t": host_np.array(optimizer.t),
        "lr": host_np.array(optimizer.get_lr()),
        "beta1": host_np.array(optimizer.beta1),
        "beta2": host_np.array(optimizer.beta2),
        "epsilon": host_np.array(optimizer.epsilon),
        "weight_decay": host_np.array(getattr(optimizer, "weight_decay", 0.0)),
        "optimizer": optimizer.__class__.__name__,
    }
    for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
        state[f"m_{i}"] = to_host_array(m)
        state[f"v_{i}"] = to_host_array(v)
    host_np.savez(path, **state)


def load_optimizer(optimizer: OptimizerType, path: Path) -> None:
    with host_np.load(path) as data:
        optimizer.t = int(data["t"])
        lrs = data["lr"].tolist()
        if isinstance(lrs, float):
            lrs = [lrs]
        optimizer.set_lr(lrs)
        if "weight_decay" in data:
            optimizer.weight_decay = float(data["weight_decay"])
        # Upload moment buffers onto the active backend (CuPy on GPU). Assigning
        # raw host arrays here would make optimizer.step() fail mixing NumPy and
        # CuPy operands, which previously broke GPU resume.
        for i in range(len(optimizer.m)):
            optimizer.m[i] = np_backend.np.asarray(data[f"m_{i}"], dtype=np_backend.dtype)
            optimizer.v[i] = np_backend.np.asarray(data[f"v_{i}"], dtype=np_backend.dtype)


def write_training_state(
    path: Path,
    *,
    step: int,
    best_val_loss: float | None,
    config: dict[str, Any],
    vocab_size: int,
) -> None:
    payload = {
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "vocab_size": vocab_size,
        "tokenizer": "tokenizer.json",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_training_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(
    ckpt_dir: Path,
    *,
    model,
    optimizer: OptimizerType,
    step: int,
    best_val_loss: float | None,
    config: dict[str, Any],
    tokenizer: FastBPETokenizer,
    rng=None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save(model, ckpt_dir / "weights.npz")
    save_optimizer(optimizer, ckpt_dir / "optimizer.npz")
    tokenizer.save(ckpt_dir / "tokenizer.json")
    write_training_state(
        ckpt_dir / "training.json",
        step=step,
        best_val_loss=best_val_loss,
        config=config,
        vocab_size=tokenizer.vocab_size,
    )
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if rng is not None:
        from toyGPT.train_utils import save_rng_state

        save_rng_state(ckpt_dir / "rng.json", rng)


def load_checkpoint(ckpt_dir: Path, *, model, optimizer: OptimizerType) -> tuple[dict[str, Any], FastBPETokenizer]:
    load(model, ckpt_dir / "weights.npz")
    load_optimizer(optimizer, ckpt_dir / "optimizer.npz")
    tokenizer = FastBPETokenizer.load(ckpt_dir / "tokenizer.json")
    return read_training_state(ckpt_dir / "training.json"), tokenizer


def resolve_resume_path(checkpoint_root: Path, resume: str) -> Path:
    if resume in ("latest", "best"):
        return checkpoint_root / resume
    path = Path(resume)
    return path if path.is_absolute() else (checkpoint_root / path).resolve()


def copy_checkpoint(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
