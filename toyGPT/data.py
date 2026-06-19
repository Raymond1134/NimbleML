"""Batch sampling for toy GPT.

The token corpus is a memmapped ``uint16`` array on the host (see
``toyGPT.fineweb``). Only the sampled batch is materialized and uploaded to the
GPU, so the full multi-GB corpus never has to fit in VRAM or RAM.
"""

from __future__ import annotations

import numpy as np  # host NumPy: the corpus and sampling live on the CPU

from NimbleML.utils.tensor import Tensor

# Re-exported so callers can do data-prep through a single module if they want.
from toyGPT.fineweb import load_token_bin, prepare_corpus  # noqa: F401


def random_batch(
    ids,
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample a random ``(inputs, targets)`` batch from a host token array.

    ``ids`` is a 1-D host array (typically a uint16 memmap). Start positions are
    drawn with a host RNG, gathered on the CPU, then uploaded to the active
    backend as int64 by ``Tensor.from_int64``.
    """
    n = int(ids.shape[0])
    row_len = seq_len + 1
    if n < row_len + 1:
        raise ValueError("Corpus too short for the requested seq_len.")

    max_start = n - row_len
    starts = rng.integers(0, max_start + 1, size=batch_size, dtype=np.int64)

    # Per-row slices from the memmap instead of 2D fancy indexing. On Windows,
    # ``ids[starts[:, None] + arange]`` over a multi-GB mmap can return garbage
    # or corrupt reads, which then surfaces as CUDA illegal memory access in
    # embedding / loss (batch_size is small, so this loop is negligible).
    windows = np.empty((batch_size, row_len), dtype=np.int64)
    for b, start in enumerate(starts):
        windows[b] = np.asarray(ids[int(start) : int(start) + row_len], dtype=np.int64)

    inputs = windows[:, :-1]
    targets = windows[:, 1:]
    return (
        Tensor.from_int64(np.ascontiguousarray(inputs).ravel(), (batch_size, seq_len)),
        Tensor.from_int64(np.ascontiguousarray(targets).ravel(), (batch_size, seq_len)),
    )
