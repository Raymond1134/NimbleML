"""Training helpers for toy GPT (optimizer groups, seeding, RNG checkpoints)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from NimbleML.utils.saveload import named_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as host_np

    host_np.random.seed(seed)


def adamw_param_groups(model, *, lr: float, weight_decay: float) -> list[dict]:
    """AdamW groups: weight decay on ``*.weights`` only (not biases / norm gamma)."""
    decay_params = []
    nodecay_params = []
    for name, param in named_parameters(model):
        if name.endswith(".weights"):
            decay_params.append(param)
        else:
            nodecay_params.append(param)
    return [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": nodecay_params, "lr": lr, "weight_decay": 0.0},
    ]


def capture_rng_state(rng) -> dict[str, Any]:
    """Best-effort RNG state capture.

    NumPy generators expose ``bit_generator.state`` as a JSON-friendly dict.
    CuPy generators (GPU backend) expose it as a method and don't round-trip
    through JSON, so we skip it there. Resume still restores weights and
    optimizer state; only the data-sampling RNG sequence is reseeded.
    """
    try:
        state = rng.bit_generator.state
    except AttributeError:
        return {}
    if callable(state):
        return {}
    return {"bit_generator": state}


def save_rng_state(path: Path, rng) -> None:
    try:
        payload = capture_rng_state(rng)
        text = json.dumps(payload, indent=2)
    except (TypeError, ValueError):
        text = "{}"
    path.write_text(text, encoding="utf-8")


def load_rng_state(path: Path, rng) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    state = payload.get("bit_generator")
    if not state:
        return
    try:
        rng.bit_generator.state = state
    except (AttributeError, ValueError, TypeError):
        pass
