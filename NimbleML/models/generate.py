"""Autoregressive generation with optional KV-cache."""
from __future__ import annotations
import numpy as host_np
from NimbleML.utils.grad_mode import no_grad
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def _sample_logits(logits_1d, *, temperature: float, top_k: int, top_p: float):
    """Sample one token id. Runs on host NumPy: the vector is vocab-sized and
    CuPy lacks ``random.choice(p=...)``."""
    get = getattr(logits_1d, "get", None)
    logits = host_np.asarray(
        get() if get is not None else logits_1d, dtype=host_np.float32
    ).reshape(-1)
    if temperature <= 0:
        return int(host_np.argmax(logits))
    logits = logits / float(temperature)
    if top_k and top_k > 0:
        k = min(top_k, logits.size)
        thresh = host_np.partition(logits, -k)[-k]
        logits = host_np.where(logits < thresh, -1e9, logits)
    probs = host_np.exp(logits - host_np.max(logits))
    probs = probs / host_np.sum(probs)
    if top_p and 0.0 < top_p < 1.0:
        order = host_np.argsort(probs)[::-1]
        sorted_p = probs[order]
        cdf = host_np.cumsum(sorted_p)
        mask = cdf > top_p
        mask[0] = False
        sorted_p = host_np.where(mask, 0.0, sorted_p)
        sorted_p = sorted_p / host_np.sum(sorted_p)
        pick = order[int(host_np.random.choice(len(sorted_p), p=sorted_p))]
        return int(pick)
    return int(host_np.random.choice(probs.size, p=probs))


def generate(
    model,
    input_ids,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    eos_id: int | None = None,
    use_kv_cache: bool = True,
):
    """Greedy / sampled generation.

    When ``use_kv_cache`` is True and the model implements ``forward_with_cache``,
    only the newest token is forwarded each step. Otherwise falls back to full
    re-forward of the growing sequence.
    """
    if isinstance(input_ids, Tensor):
        ids = np.asarray(input_ids.data, dtype=np.int64).reshape(input_ids.shape)
    else:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)

    batch = ids.shape[0]
    generated = [list(ids[b].tolist()) for b in range(batch)]
    cache = None

    with no_grad():
        for _ in range(max_new_tokens):
            if use_kv_cache and hasattr(model, "forward_with_cache"):
                if cache is None:
                    cur = Tensor.from_int64(np.asarray(generated, dtype=np.int64).ravel(), (batch, len(generated[0])))
                    logits, cache = model.forward_with_cache(cur, cache=None)
                else:
                    last = np.asarray([[generated[b][-1] for b in range(batch)]], dtype=np.int64).T
                    cur = Tensor.from_int64(last.ravel(), (batch, 1))
                    logits, cache = model.forward_with_cache(cur, cache=cache)
            else:
                cur = Tensor.from_int64(np.asarray(generated, dtype=np.int64).ravel(), (batch, len(generated[0])))
                logits = model.forward(cur)

            logits_arr = np.asarray(logits.data).reshape(logits.shape)
            for b in range(batch):
                tok = _sample_logits(logits_arr[b, -1], temperature=temperature, top_k=top_k, top_p=top_p)
                generated[b].append(tok)
            if eos_id is not None and all(generated[b][-1] == eos_id for b in range(batch)):
                break

    return np.asarray(generated, dtype=np.int64)
