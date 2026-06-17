"""Fast byte-level BPE tokenizer for toyGPT (HuggingFace ``tokenizers``, Rust).

A thin wrapper over a ``tokenizers.Tokenizer`` that exposes the small interface
the rest of toyGPT needs (``vocab_size``, ``encode``/``encode_batch``,
``decode``, ``save``/``load``). It trains and encodes orders of magnitude faster
than a pure-Python BPE, which is what makes tokenizing ~1B tokens feasible.

The on-disk format is the native ``tokenizers`` ``tokenizer.json`` so it can be
saved into checkpoints and reloaded for inference without any conversion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence, Union

EOT_TOKEN = "<|endoftext|>"


class FastBPETokenizer:
    """Byte-level BPE backed by the Rust ``tokenizers`` library."""

    def __init__(self, tok) -> None:
        self._tok = tok
        eot = tok.token_to_id(EOT_TOKEN)
        # Fall back to 0 only if a tokenizer was built without the special
        # token; all tokenizers this module trains include it.
        self.eot_id = int(eot) if eot is not None else 0

    @classmethod
    def train(
        cls,
        text_iter: Iterable[str],
        *,
        vocab_size: int,
        save_path: Union[str, Path, None] = None,
        min_frequency: int = 2,
        show_progress: bool = False,
    ) -> "FastBPETokenizer":
        """Train a byte-level BPE on ``text_iter`` and (optionally) save it."""
        from tokenizers import ByteLevelBPETokenizer, Tokenizer

        bl = ByteLevelBPETokenizer()
        bl.train_from_iterator(
            text_iter,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=[EOT_TOKEN],
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            bl.save(str(save_path))
            tok = Tokenizer.from_file(str(save_path))
        else:
            # Round-trip through a JSON string to get a stable base Tokenizer.
            tok = Tokenizer.from_str(bl._tokenizer.to_str())
        return cls(tok)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FastBPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    def fingerprint(self) -> str:
        """Stable content hash (token->id map), independent of JSON formatting.

        Used to key the encoded-corpus cache so resuming reuses the bins even
        when a checkpoint re-serializes ``tokenizer.json`` with cosmetic diffs.
        """
        items = sorted(self._tok.get_vocab().items(), key=lambda kv: kv[1])
        payload = json.dumps(items, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def encode(self, text: str, **_kwargs) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode many strings at once (parallelized across CPU cores)."""
        return [enc.ids for enc in self._tok.encode_batch(list(texts))]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        int_ids = [int(i) for i in ids]
        if not int_ids:
            return ""
        return self._tok.decode(int_ids, skip_special_tokens=skip_special_tokens)
