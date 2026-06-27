"""NimbleML GPT train-step helpers shared by benchmark scripts."""
from __future__ import annotations

import numpy as host_np

from NimbleML.models.gpt import GPT
from NimbleML.optimizers import AdamW
from NimbleML.utils.clip_grad import clip_grad_norm_
from NimbleML.utils.tensor import Tensor

from .config import ReferenceConfig


def make_inputs(cfg: ReferenceConfig) -> tuple[Tensor, Tensor, float]:
    inputs = Tensor.from_int64(
        host_np.random.randint(0, cfg.vocab, size=(cfg.batch, cfg.seq), dtype=host_np.int64).ravel(),
        (cfg.batch, cfg.seq),
    )
    targets = Tensor.from_int64(
        host_np.random.randint(0, cfg.vocab, size=(cfg.batch, cfg.seq), dtype=host_np.int64).ravel(),
        (cfg.batch, cfg.seq),
    )
    return inputs, targets, cfg.tokens_per_step


def make_model(
    cfg: ReferenceConfig,
    *,
    fused_blocks: bool = True,
    fused_trunk: bool = False,
) -> GPT:
    return GPT(
        cfg.vocab,
        cfg.d_model,
        cfg.heads,
        cfg.layers,
        cfg.seq,
        ff_mult=cfg.ff_mult,
        fused_blocks=fused_blocks,
        fused_trunk=fused_trunk,
    )


def zero_grads(params, *, set_to_none: bool = True) -> None:
    for param in params:
        if set_to_none:
            param.grad = None
        else:
            param.zero_grad()


def build_train_step(model: GPT, inputs: Tensor, targets: Tensor, cfg: ReferenceConfig):
    opt = AdamW(model.parameters(), learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

    def train_step() -> None:
        opt.zero_grad(set_to_none=True)
        loss = model.compute_loss(inputs, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        model.clear_pos_encoding_cache()

    return train_step
