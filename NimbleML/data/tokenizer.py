"""Byte-level BPE tokenizer."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

BYTE_VOCAB_SIZE = 256
_NO_MERGE_RANK = np.iinfo(np.int32).max
_BINCOUNT_MAX_TABLE = 8_000_000  # cap pair histogram size (~32MB int32)


def _byte_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}


def _text_to_ids(text: str) -> np.ndarray:
    """UTF-8 text -> int32 token ids via zero-copy byte buffer."""
    data = text.encode("utf-8")
    if not data:
        return np.empty(0, dtype=np.int32)
    return np.frombuffer(data, dtype=np.uint8).astype(np.int32, copy=False)


def _merge_all_pair(ids: np.ndarray, pair: tuple[int, int], new_id: int) -> np.ndarray:
    """Merge every non-overlapping occurrence of ``pair`` (vectorized, left-to-right)."""
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


def _pair_counts(ids: np.ndarray) -> Counter:
    """Count adjacent token pairs using vectorized histogram."""
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
        nz = np.flatnonzero(hist)
        for key in nz:
            out[(int(key // stride), int(key % stride))] = int(hist[key])
        return out

    unique, counts = np.unique(keys, return_counts=True)
    for key, count in zip(unique, counts):
        out[(int(key // stride), int(key % stride))] = int(count)
    return out


def _pick_best_pair(counts: Counter) -> tuple[int, int] | None:
    if not counts:
        return None
    max_count = max(counts.values())
    candidates = [pair for pair, count in counts.items() if count == max_count]
    return min(candidates)


def _pair_repr(pair: tuple[int, int], vocab: dict[int, bytes]) -> str:
    a, b = pair
    try:
        return f"{vocab[a]!r}+{vocab[b]!r}"
    except KeyError:
        return f"({a},{b})"


class BPETokenizer:
    """Byte-level BPE tokenizer (UTF-8 bytes, GPT-2 style merge order)."""

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = _byte_vocab()
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._vocab_bytes: list[bytes] = [bytes([i]) for i in range(BYTE_VOCAB_SIZE)]
        self._rank_lut = np.full((BYTE_VOCAB_SIZE, BYTE_VOCAB_SIZE), _NO_MERGE_RANK, dtype=np.int32)
        self._train_corpus_ids: np.ndarray | None = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _rebuild_vocab_and_ranks(self) -> None:
        self.vocab = _byte_vocab()
        for i, pair in enumerate(self.merges):
            new_id = BYTE_VOCAB_SIZE + i
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        size = len(self.vocab)
        self._vocab_bytes = [self.vocab[i] for i in range(size)]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        lut = np.full((size, size), _NO_MERGE_RANK, dtype=np.int32)
        for rank, (a, b) in enumerate(self.merges):
            lut[a, b] = rank
        self._rank_lut = lut

    def _best_merge(self, ids: np.ndarray) -> tuple[int, int, int] | None:
        """Return (rank, a, b) for the lowest-rank mergeable pair in ``ids``."""
        if ids.size < 2:
            return None

        left = ids[:-1]
        right = ids[1:]
        ranks = self._rank_lut[left, right]
        best_rank = int(ranks.min())
        if best_rank == _NO_MERGE_RANK:
            return None

        idx = int(ranks.argmin())
        return best_rank, int(left[idx]), int(right[idx])

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
        *,
        max_train_chars: int | None = None,
        log_every: int = 1,
        **kwargs,
    ) -> "BPETokenizer":
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be >= {BYTE_VOCAB_SIZE} for byte-level BPE.")

        raw_len = len(text)
        if max_train_chars is not None and max_train_chars > 0:
            text = text[:max_train_chars]

        ids = _text_to_ids(text)
        if ids.size == 0:
            raise ValueError("Training text is empty after UTF-8 encoding.")

        target_merges = vocab_size - BYTE_VOCAB_SIZE
        self.merges = []
        self._train_corpus_ids = None

        if verbose:
            print(
                f"[bpe] start train | target_vocab={vocab_size} merges={target_merges} "
                f"corpus_chars={len(text):,} corpus_tokens={ids.size:,}"
                + (f" (capped from {raw_len:,})" if len(text) < raw_len else "")
            )

        train_start = time.perf_counter()
        running_vocab = dict(self.vocab)

        for merge_idx in range(target_merges):
            t0 = time.perf_counter()
            counts = _pair_counts(ids)
            t_count = time.perf_counter() - t0

            best_pair = _pick_best_pair(counts)
            if best_pair is None:
                if verbose:
                    print(f"[bpe] merge {merge_idx}: no pairs left, stopping early")
                break

            pair_count = counts[best_pair]
            new_id = BYTE_VOCAB_SIZE + merge_idx
            self.merges.append(best_pair)
            running_vocab[new_id] = running_vocab[best_pair[0]] + running_vocab[best_pair[1]]

            t1 = time.perf_counter()
            ids = _merge_all_pair(ids, best_pair, new_id)
            t_merge = time.perf_counter() - t1

            should_log = verbose and (
                log_every <= 1 or (merge_idx + 1) % log_every == 0 or merge_idx == 0 or merge_idx + 1 == target_merges
            )
            if should_log:
                pair_label = _pair_repr(best_pair, running_vocab)
                print(
                    f"[bpe] merge {merge_idx + 1:5d}/{target_merges} | pair={pair_label} "
                    f"count={pair_count:,} -> id={new_id} | "
                    f"tokens={ids.size:,} | count_ms={t_count * 1000:.1f} merge_ms={t_merge * 1000:.1f}"
                )

        self._train_corpus_ids = ids.copy()
        self._rebuild_vocab_and_ranks()

        if verbose:
            elapsed = time.perf_counter() - train_start
            print(
                f"[bpe] done | vocab={self.vocab_size} merges={len(self.merges)} "
                f"final_tokens={ids.size:,} elapsed={elapsed:.2f}s"
            )
        return self

    def take_train_corpus_ids(self) -> list[int] | None:
        """Return merged training corpus ids if ``train()`` just ran (avoids re-encoding)."""
        if self._train_corpus_ids is None:
            return None
        ids = self._train_corpus_ids.tolist()
        self._train_corpus_ids = None
        return ids

    def _encode_ids(self, ids: np.ndarray, *, verbose: bool = False) -> np.ndarray:
        if not self.merges:
            return ids

        steps = 0
        encode_start = time.perf_counter()
        while ids.size >= 2:
            best = self._best_merge(ids)
            if best is None:
                break
            rank, a, b = best
            ids = _merge_all_pair(ids, (a, b), BYTE_VOCAB_SIZE + rank)
            steps += 1

        if verbose:
            elapsed = time.perf_counter() - encode_start
            print(f"[bpe] encode | tokens_out={ids.size:,} merge_steps={steps} elapsed={elapsed:.3f}s")
        return ids

    def encode(self, text: str, verbose: bool = False, **kwargs) -> list[int]:
        ids = _text_to_ids(text)
        return self._encode_ids(ids, verbose=verbose).tolist()

    def encode_batch(self, texts: list[str], verbose: bool = False) -> list[list[int]]:
        if verbose:
            print(f"[bpe] encode_batch | n={len(texts)}")
        return [self.encode(text, verbose=False) for text in texts]

    def decode(self, ids: list[int] | np.ndarray) -> str:
        if isinstance(ids, np.ndarray):
            if ids.size == 0:
                return ""
            id_list = ids.tolist()
        else:
            if not ids:
                return ""
            id_list = ids

        try:
            return b"".join(self._vocab_bytes[i] for i in id_list).decode("utf-8", errors="replace")
        except IndexError as exc:
            raise KeyError("Unknown token id in decode input") from exc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nimbleml_byte_bpe_v1",
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "nimbleml_byte_bpe_v1":
            raise ValueError(f"Unsupported tokenizer format in {path}")

        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._rebuild_vocab_and_ranks()
        return tokenizer
