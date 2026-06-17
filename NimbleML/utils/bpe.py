"""Byte-level BPE primitives (NumPy)."""

from __future__ import annotations

from collections import Counter

import numpy as np

BYTE_VOCAB_SIZE = 256
NO_MERGE_RANK = np.iinfo(np.int32).max
_BINCOUNT_MAX_TABLE = 8_000_000


def byte_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}


def utf8_to_ids(text: str) -> np.ndarray:
    """UTF-8 text -> int32 token ids (one id per byte)."""
    data = text.encode("utf-8")
    if not data:
        return np.empty(0, dtype=np.int32)
    return np.frombuffer(data, dtype=np.uint8).astype(np.int32, copy=False)


def merge_all_pair(ids: np.ndarray, pair: tuple[int, int], new_id: int) -> np.ndarray:
    """Merge every non-overlapping occurrence of ``pair`` (left-to-right)."""
    a, b = int(pair[0]), int(pair[1])
    new_id = int(new_id)
    if ids.size < 2:
        return ids

    match = (ids[:-1] == a) & (ids[1:] == b)
    if not np.any(match):
        return ids

    idx = np.flatnonzero(match)
    if idx.size > 1:
        keep = np.ones(idx.size, dtype=bool)
        keep[1:] = idx[1:] - idx[:-1] > 1
        idx = idx[keep]

    drop = np.zeros(ids.size, dtype=bool)
    drop[idx + 1] = True
    out = ids.copy()
    out[idx] = new_id
    return out[~drop]


def pair_counts(ids: np.ndarray) -> Counter:
    """Count adjacent token pairs using a vectorized histogram."""
    if ids.size < 2:
        return Counter()

    left = ids[:-1].astype(np.int64, copy=False)
    right = ids[1:].astype(np.int64, copy=False)
    stride = int(max(left.max(initial=0), right.max(initial=0))) + 1
    keys = left * stride + right
    table_size = stride * stride

    out: Counter = Counter()
    if table_size <= _BINCOUNT_MAX_TABLE:
        hist = np.bincount(keys, minlength=table_size)
        for key in np.flatnonzero(hist):
            out[(int(key // stride), int(key % stride))] = int(hist[key])
        return out

    unique, counts = np.unique(keys, return_counts=True)
    for key, count in zip(unique, counts):
        out[(int(key // stride), int(key % stride))] = int(count)
    return out
