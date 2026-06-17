"""Text sampling from a trained toy GPT checkpoint."""

from __future__ import annotations

import gc

from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor

import numpy as _host_np


def prompt_ids_from_corpus(corpus, seq_len: int) -> list[int]:
    """First ``seq_len`` token ids from a host or device corpus array."""
    slice_arr = corpus[:seq_len]
    if hasattr(slice_arr, "get"):
        slice_arr = slice_arr.get()
    return [int(x) for x in np.asarray(slice_arr, dtype=np.int64).tolist()]


def sample_text(
    model,
    tokenizer,
    *,
    prompt_ids: list[int],
    seq_len: int,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 0,
) -> str:
    """Autoregressively extend ``prompt_ids`` and decode to text."""
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = ids[-seq_len:]
        if not window:
            break
        if len(window) < seq_len:
            window = [0] * (seq_len - len(window)) + window
        inputs = Tensor.from_int64(window, (1, seq_len))
        logits = model.forward(inputs)
        logits_arr = np.asarray(logits.data, dtype=np.float32).reshape(seq_len, -1)
        last = logits_arr[-1]
        # Each forward builds an autograd graph with reference cycles; drop it
        # so generating many tokens does not accumulate GPU memory.
        del logits
        gc.collect()
        if temperature <= 0:
            next_id = int(np.argmax(last))
        else:
            scaled = last / temperature
            if top_k > 0 and top_k < scaled.size:
                top_idx = np.argpartition(scaled, -top_k)[-top_k:]
                keep = np.full_like(scaled, -np.inf)
                keep[top_idx] = scaled[top_idx]
                scaled = keep
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= np.sum(probs)
            if hasattr(probs, "get"):
                probs = probs.get()
            probs_host = _host_np.clip(_host_np.asarray(probs, dtype=_host_np.float64), 0.0, None)
            total = probs_host.sum()
            if not _host_np.isfinite(total) or total <= 0:
                last_host = last.get() if hasattr(last, "get") else last
                next_id = int(_host_np.argmax(_host_np.asarray(last_host)))
            else:
                next_id = int(_host_np.random.choice(len(probs_host), p=probs_host / total))
        ids.append(next_id)
    return tokenizer.decode(ids)
