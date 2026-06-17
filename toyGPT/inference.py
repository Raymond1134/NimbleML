"""Load a trained toy GPT checkpoint for inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

from NimbleML.utils.saveload import load
from toyGPT.checkpoint import load_checkpoint, read_training_state, resolve_resume_path
from toyGPT.config import ToyGPTConfig


def _model_hparams(cfg: ToyGPTConfig, state: dict) -> dict:
    saved = dict(state.get("config") or {})
    ckpt_vocab = int(state.get("vocab_size", saved.get("vocab_size", cfg.vocab_size)))
    return {
        "vocab_size": ckpt_vocab,
        "d_model": int(saved.get("d_model", cfg.d_model)),
        "n_head": int(saved.get("n_head", cfg.n_head)),
        "n_layer": int(saved.get("n_layer", cfg.n_layer)),
        "seq_len": int(saved.get("seq_len", cfg.seq_len)),
        "ff_mult": int(saved.get("ff_mult", cfg.ff_mult)),
    }


def load_tokenizer(ckpt_dir: Path, cfg: ToyGPTConfig):
    from toyGPT.fast_tokenizer import FastBPETokenizer

    ckpt_tok = ckpt_dir / "tokenizer.json"
    if ckpt_tok.is_file():
        return FastBPETokenizer.load(ckpt_tok)
    if cfg.tokenizer_path.is_file():
        return FastBPETokenizer.load(cfg.tokenizer_path)
    raise FileNotFoundError(
        f"No tokenizer at {ckpt_tok} or {cfg.tokenizer_path}. "
        "Train first or copy tokenizer.json into the checkpoint folder."
    )


def load_for_inference(cfg: ToyGPTConfig, checkpoint: str = "best"):
    """Return ``(model, tokenizer, training_state, ckpt_dir)`` ready for sampling."""
    from NimbleML.models import GPT
    from NimbleML.optimizers import AdamW
    from NimbleML.utils.np_backend import apply_runtime_config

    ckpt_dir = resolve_resume_path(cfg.checkpoint_dir, checkpoint)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_dir}\n"
            "Train first: python toyGPT\\train_gpt.py"
        )

    state = read_training_state(ckpt_dir / "training.json")
    config_json = ckpt_dir / "config.json"
    if config_json.is_file() and not state.get("config"):
        state["config"] = json.loads(config_json.read_text(encoding="utf-8"))

    saved = dict(state.get("config") or {})
    runtime_device = str(saved.get("device", cfg.device))
    runtime_dtype = str(saved.get("dtype", cfg.dtype))
    os.environ["NIMBLEML_DEVICE"] = runtime_device
    os.environ["NIMBLEML_DTYPE"] = runtime_dtype

    hp = _model_hparams(cfg, state)
    apply_runtime_config(runtime_device, runtime_dtype)

    tokenizer = load_tokenizer(ckpt_dir, cfg)
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size != hp["vocab_size"]:
        hp["vocab_size"] = vocab_size

    model = GPT(
        hp["vocab_size"],
        hp["d_model"],
        hp["n_head"],
        hp["n_layer"],
        hp["seq_len"],
        ff_mult=hp["ff_mult"],
    )
    optimizer = AdamW(model.parameters(), learning_rate=cfg.lr)
    weights = ckpt_dir / "weights.npz"
    if weights.is_file():
        load(model, weights)
    else:
        load_checkpoint(ckpt_dir, model=model, optimizer=optimizer)
    model.clear_pos_encoding_cache()

    return model, tokenizer, state, ckpt_dir
