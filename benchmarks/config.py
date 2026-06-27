"""Pinned benchmark configuration — change only deliberately; re-run after perf work."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceConfig:
    """Standard GPT benchmark shape (default for all benchmarks/ scripts)."""

    vocab: int = 4096
    d_model: int = 512
    heads: int = 8
    layers: int = 8
    ff_mult: int = 4
    seq: int = 256
    batch: int = 4
    warmup: int = 3
    runs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    @property
    def tokens_per_step(self) -> float:
        return float(self.batch * self.seq)

    @property
    def d_k(self) -> int:
        return self.d_model // self.heads


# Re-run benchmarks after Phase 1–3 changes against this config.
REFERENCE = ReferenceConfig()

# Smaller shape for quick local smoke / CPU runs (`--quick` CLI flag).
QUICK = ReferenceConfig(
    vocab=256,
    d_model=128,
    heads=4,
    layers=2,
    seq=32,
    batch=8,
    warmup=2,
    runs=5,
)
