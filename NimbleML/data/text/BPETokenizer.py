"""Byte-level BPE tokenizer."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from NimbleML.utils.bpe import (BYTE_VOCAB_SIZE, NO_MERGE_RANK, byte_vocab, merge_all_pair, pair_counts, utf8_to_ids)

__all__ = ["BYTE_VOCAB_SIZE", "BPETokenizer"]


class BPETokenizer:
    """Byte-level Byte Pair Encoding (BPE) tokenizer.

    Learns merge rules from UTF-8 encoded text and uses them to encode text
    into token IDs and decode token IDs back into text. The tokenizer begins
    with a vocabulary consisting of all 256 byte values and iteratively
    adds merged byte-pair tokens during training.
    """
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {}
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._vocab_bytes: list[bytes] = []
        self._rank_lut = np.empty((0, 0), dtype=np.int32)
        self._rebuild_vocab_and_ranks()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

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
        """Trains the tokenizer on a text corpus.

        Learns BPE merge rules by repeatedly merging the most frequent adjacent
        token pair until the target vocabulary size is reached.

        Args:
            text (str): Training corpus.
            vocab_size (int): Desired vocabulary size, including the initial byte vocabulary.
            verbose (bool): Whether to print training progress.
            max_train_chars (int, optional): Maximum number of characters from the corpus to use for training.
            log_every (int): Interval between progress updates when verbose.

        Returns:
            BPETokenizer: The trained tokenizer.

        Raises:
            ValueError: If vocab_size is smaller than the byte vocabulary size or the training corpus is empty.
        
        Examples:
            >>> text = "hello hello hello world"
            >>> tokenizer = BPETokenizer()
            >>> tokenizer.train(text, vocab_size=300, verbose=False)
        """
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be >= {BYTE_VOCAB_SIZE} for byte-level BPE.")

        raw_len = len(text)
        if max_train_chars is not None and max_train_chars > 0:
            text = text[:max_train_chars]

        ids = utf8_to_ids(text)
        if ids.size == 0:
            raise ValueError("Training text is empty after UTF-8 encoding.")

        target_merges = vocab_size - BYTE_VOCAB_SIZE
        self.merges = []

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
            counts = pair_counts(ids)
            t_count = time.perf_counter() - t0

            if not counts:
                if verbose:
                    print(f"[bpe] merge {merge_idx}: no pairs left, stopping early")
                break

            max_count = max(counts.values())
            best_pair = min(pair for pair, count in counts.items() if count == max_count)

            pair_count = counts[best_pair]
            new_id = BYTE_VOCAB_SIZE + merge_idx
            self.merges.append(best_pair)
            running_vocab[new_id] = running_vocab[best_pair[0]] + running_vocab[best_pair[1]]

            t1 = time.perf_counter()
            ids = merge_all_pair(ids, best_pair, new_id)
            t_merge = time.perf_counter() - t1

            should_log = verbose and (log_every <= 1 or (merge_idx + 1) % log_every == 0 or merge_idx == 0 or merge_idx + 1 == target_merges)
            if should_log:
                a, b = best_pair
                try:
                    pair_label = f"{running_vocab[a]!r}+{running_vocab[b]!r}"
                except KeyError:
                    pair_label = f"({a},{b})"
                print(
                    f"[bpe] merge {merge_idx + 1:5d}/{target_merges} | pair={pair_label} "
                    f"count={pair_count:,} -> id={new_id} | "
                    f"tokens={ids.size:,} | count_ms={t_count * 1000:.1f} merge_ms={t_merge * 1000:.1f}"
                )

        self._rebuild_vocab_and_ranks()

        if verbose:
            elapsed = time.perf_counter() - train_start
            print(
                f"[bpe] done | vocab={self.vocab_size} merges={len(self.merges)} "
                f"final_tokens={ids.size:,} elapsed={elapsed:.2f}s"
            )
        return self

    def encode(self, text: str, verbose: bool = False, log_every: int = 1, label: str = "", **kwargs) -> list[int]:
        """Encodes text into token IDs.

        Args:
            text (str): Input text.
            verbose (bool): Whether to print encoding progress.
            log_every (int): Interval between progress updates when verbose.
            label (str): Optional label used in verbose logging.

        Returns:
            list[int]: Encoded token IDs.
        
        Examples:
            >>> text = "hello world"
            >>> ids = tokenizer.encode(text, verbose=False)
        """
        return self._encode_ids(utf8_to_ids(text), verbose=verbose, log_every=log_every, label=label).tolist()

    def encode_batch(self, texts: list[str], verbose: bool = False) -> list[list[int]]:
        """Encodes a list of text strings into token IDs.

        Args:
            texts (list[str]): List of input text strings.
            verbose (bool): Whether to print encoding progress.
        
        Returns:
            list[list[int]]: List of encoded token IDs for each input text.
        
        Examples:
            >>> texts = ["hello world", "goodbye!"]
            >>> ids = tokenizer.encode_batch(texts, verbose=False)
        """
        if verbose:
            print(f"[bpe] encode_batch | n={len(texts)}")
        return [self.encode(text, verbose=False) for text in texts]

    def decode(self, ids: list[int] | np.ndarray) -> str:
        """Decodes a list of token IDs back into a string.
        
        Args:
            ids (list[int] | np.ndarray): List of token IDs.
        
        Returns:
            str: Decoded string.
        
        Examples:
            >>> ids = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
            >>> text = tokenizer.decode(ids)
        """
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
        """Saves the tokenizer to a file.
        
        Args:
            path (str | Path): Path to save the tokenizer.
        
        Examples:
            >>> tokenizer = BPETokenizer()
            >>> tokenizer.save("tokenizer.json")
        """
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
        """Loads the tokenizer from a file.

        Args:
            path (str | Path): Path to load the tokenizer from.

        Returns:
            BPETokenizer: The loaded tokenizer.

        Examples:
            >>> tokenizer = BPETokenizer.load("tokenizer.json")
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "nimbleml_byte_bpe_v1":
            raise ValueError(f"Unsupported tokenizer format in {path}")

        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._rebuild_vocab_and_ranks()
        return tokenizer

    def _rebuild_vocab_and_ranks(self) -> None:
        self.vocab = byte_vocab()
        for i, pair in enumerate(self.merges):
            new_id = BYTE_VOCAB_SIZE + i
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        size = len(self.vocab)
        self._vocab_bytes = [self.vocab[i] for i in range(size)]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        lut = np.full((size, size), NO_MERGE_RANK, dtype=np.int32)
        for rank, (a, b) in enumerate(self.merges):
            lut[a, b] = rank
        self._rank_lut = lut

    def _best_merge(self, ids: np.ndarray) -> tuple[int, int, int] | None:
        if ids.size < 2:
            return None

        left = ids[:-1]
        right = ids[1:]
        ranks = self._rank_lut[left, right]
        best_rank = int(ranks.min())
        if best_rank == NO_MERGE_RANK:
            return None

        idx = int(ranks.argmin())
        return best_rank, int(left[idx]), int(right[idx])

    def _encode_ids(
        self,
        ids: np.ndarray,
        *,
        verbose: bool = False,
        log_every: int = 1,
        label: str = "",
    ) -> np.ndarray:
        if not self.merges:
            return ids

        steps = 0
        encode_start = time.perf_counter()
        tag = f"{label} " if label else ""
        target_steps = len(self.merges)

        if verbose:
            print(
                f"[bpe] encode {tag}start | corpus_tokens={ids.size:,} "
                f"merges={target_steps} log_every={log_every}"
            )

        while ids.size >= 2:
            t0 = time.perf_counter()
            best = self._best_merge(ids)
            if best is None:
                break
            rank, a, b = best
            ids = merge_all_pair(ids, (a, b), BYTE_VOCAB_SIZE + rank)
            steps += 1
            step_ms = (time.perf_counter() - t0) * 1000.0

            should_log = verbose and (
                log_every <= 1
                or steps % log_every == 0
                or steps == 1
                or steps == target_steps
            )
            if should_log:
                print(
                    f"[bpe] encode {tag}step {steps:5d}/{target_steps} | "
                    f"tokens={ids.size:,} | step_ms={step_ms:.1f}"
                )

        if verbose:
            elapsed = time.perf_counter() - encode_start
            print(
                f"[bpe] encode {tag}done | tokens_out={ids.size:,} "
                f"merge_steps={steps} elapsed={elapsed:.2f}s"
            )
        return ids
