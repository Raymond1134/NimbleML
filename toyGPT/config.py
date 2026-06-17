"""Load toy GPT training config from TOML."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


TOYGPT_ROOT = Path(__file__).resolve().parent


def _parse_toml(text: str) -> dict[str, dict[str, Any]]:
    """Minimal TOML parser for flat [section] key = value tables."""
    data: dict[str, dict[str, Any]] = {}
    section = ""
    _number = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if value.startswith('"') and value.endswith('"'):
            parsed: Any = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            parsed = value[1:-1]
        elif value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        elif _number.match(value):
            parsed = float(value) if ("." in value or "e" in value.lower()) else int(value)
        else:
            parsed = value
        data.setdefault(section, {})[key] = parsed
    return data


def load_toml(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib

        return tomllib.loads(text)
    except ImportError:
        return _parse_toml(text)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class ToyGPTConfig:
    vocab_size: int = 8192
    d_model: int = 320
    n_layer: int = 5
    n_head: int = 5
    seq_len: int = 256
    ff_mult: int = 4
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_clip: float = 1.0
    dtype: str = "float32"
    device: str = "gpu"
    checkpoint_every: int = 500
    checkpoint_dir: Path = TOYGPT_ROOT / "checkpoints"
    dataset: str = "fineweb-edu"
    hf_repo: str = "HuggingFaceFW/fineweb-edu"
    hf_subset: str = "sample-10BT"
    train_tokens: int = 1_300_000_000
    val_tokens: int = 1_000_000
    tokenizer_sample_mb: int = 400
    data_dir: Path = TOYGPT_ROOT / "data"
    cache_dir: Path = TOYGPT_ROOT / "data" / "cache"
    tokenizer_path: Path = TOYGPT_ROOT / "data" / "tokenizer.json"
    eval_every: int = 200
    eval_batches: int = 20
    sample_chars: int = 200
    temperature: float = 0.8
    verbose: int = 1
    log_every: int = 1
    bpe_log_every: int = 1
    rolling_avg: int = 50
    early_stop_patience: int = 0
    log_grad_norm: int = 0
    seed: int = 42
    config_path: Path = TOYGPT_ROOT / "gpt_toy_config.toml"

    @classmethod
    def from_toml(cls, path: Path) -> "ToyGPTConfig":
        raw = load_toml(path)
        model = raw.get("model", {})
        train = raw.get("train", {})
        data = raw.get("data", {})
        eval_cfg = raw.get("eval", {})
        log_cfg = raw.get("log", {})
        seed_cfg = raw.get("seed", {})

        kwargs = {
            "vocab_size": int(model.get("vocab_size", 0)),
            "d_model": int(model.get("d_model", 320)),
            "n_layer": int(model.get("n_layer", 5)),
            "n_head": int(model.get("n_head", 5)),
            "seq_len": int(model.get("seq_len", 256)),
            "ff_mult": int(model.get("ff_mult", 4)),
            "batch_size": int(train.get("batch_size", 16)),
            "lr": float(train.get("lr", 3e-4)),
            "weight_decay": float(train.get("weight_decay", 0.1)),
            "warmup_steps": int(train.get("warmup_steps", 100)),
            "max_steps": int(train.get("max_steps", 5000)),
            "grad_clip": float(train.get("grad_clip", 1.0)),
            "dtype": str(train.get("dtype", "float32")),
            "device": str(train.get("device", "gpu")),
            "checkpoint_every": int(train.get("checkpoint_every", 500)),
            "checkpoint_dir": _resolve_path(train.get("checkpoint_dir", "checkpoints"), base=TOYGPT_ROOT),
            "dataset": str(data.get("dataset", "fineweb-edu")),
            "hf_repo": str(data.get("hf_repo", "HuggingFaceFW/fineweb-edu")),
            "hf_subset": str(data.get("hf_subset", "sample-10BT")),
            "train_tokens": int(data.get("train_tokens", 1_300_000_000)),
            "val_tokens": int(data.get("val_tokens", 1_000_000)),
            "tokenizer_sample_mb": int(data.get("tokenizer_sample_mb", 400)),
            "data_dir": _resolve_path(data.get("data_dir", "data"), base=TOYGPT_ROOT),
            "cache_dir": _resolve_path(data.get("cache_dir", "data/cache"), base=TOYGPT_ROOT),
            "tokenizer_path": _resolve_path(data.get("tokenizer_path", "data/tokenizer.json"), base=TOYGPT_ROOT),
            "eval_every": int(eval_cfg.get("eval_every", 200)),
            "eval_batches": int(eval_cfg.get("eval_batches", 20)),
            "sample_chars": int(eval_cfg.get("sample_chars", 200)),
            "temperature": float(eval_cfg.get("temperature", 0.8)),
            "verbose": int(log_cfg.get("verbose", 1)),
            "log_every": int(log_cfg.get("log_every", 1)),
            "bpe_log_every": int(log_cfg.get("bpe_log_every", 1)),
            "rolling_avg": int(log_cfg.get("rolling_avg", 50)),
            "early_stop_patience": int(train.get("early_stop_patience", 0)),
            "log_grad_norm": int(log_cfg.get("log_grad_norm", 0)),
            "seed": int(seed_cfg.get("seed", 42)),
            "config_path": path.resolve(),
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Path):
                out[field.name] = str(value)
            else:
                out[field.name] = value
        return out
