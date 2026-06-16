"""Byte-level BPE tokenizer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BYTE_VOCAB_SIZE = 256


def _byte_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}


def _merge_pair_in_list(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """Byte-level BPE tokenizer (UTF-8 bytes, GPT-2 style merge order)."""

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = _byte_vocab()
        self._merge_ranks: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _rebuild_vocab_and_ranks(self) -> None:
        self.vocab = _byte_vocab()
        for i, pair in enumerate(self.merges):
            new_id = BYTE_VOCAB_SIZE + i
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @staticmethod
    def _pair_counts(ids: list[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _pick_best_pair(counts: Counter) -> tuple[int, int] | None:
        if not counts:
            return None
        max_count = max(counts.values())
        candidates = [pair for pair, count in counts.items() if count == max_count]
        return min(candidates)

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
        *,
        max_train_chars: int | None = None,
        **kwargs,
    ) -> "BPETokenizer":
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be >= {BYTE_VOCAB_SIZE} for byte-level BPE.")

        if max_train_chars is not None and max_train_chars > 0:
            text = text[:max_train_chars]

        ids = list(text.encode("utf-8"))
        if not ids:
            raise ValueError("Training text is empty after UTF-8 encoding.")

        target_merges = vocab_size - BYTE_VOCAB_SIZE
        self.merges = []

        for merge_idx in range(target_merges):
            counts = self._pair_counts(ids)
            best_pair = self._pick_best_pair(counts)
            if best_pair is None:
                break

            new_id = BYTE_VOCAB_SIZE + merge_idx
            self.merges.append(best_pair)
            ids = _merge_pair_in_list(ids, best_pair, new_id)

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"BPE merge {merge_idx + 1}/{target_merges}: {best_pair} -> {new_id}")

        self._rebuild_vocab_and_ranks()
        if verbose:
            print(f"BPE training done: {self.vocab_size} tokens ({len(self.merges)} merges)")
        return self

    def encode(self, text: str, verbose: bool = False, **kwargs) -> list[int]:
        if not self.merges:
            return list(text.encode("utf-8"))

        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None or best_rank is None:
                break
            ids = _merge_pair_in_list(ids, best_pair, BYTE_VOCAB_SIZE + best_rank)
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        pieces = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise KeyError(f"Unknown token id {token_id}")
            pieces.append(self.vocab[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

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
