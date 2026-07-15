"""Attention mask helpers."""
from functools import lru_cache
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


@lru_cache(maxsize=64)
def _causal_mask_cached(seq_len: int, device: str, dtype_name: str):
    mask_arr = np.triu(
        np.full((seq_len, seq_len), -np.inf, dtype=np_backend.dtype), k=1
    )
    return Tensor(mask_arr.ravel(), mask_arr.shape, requires_grad=False)


def make_causal_mask(seq_len):
    """Upper-triangular (seq_len, seq_len) additive mask for causal attention."""
    return causal_mask_tensor(seq_len)._view()


def causal_mask_tensor(seq_len):
    """Cached (seq_len, seq_len) additive causal mask as a no-grad Tensor.

    Keyed by (seq_len, device, dtype) so backend/dtype switches never return a
    stale array from another backend.
    """
    return _causal_mask_cached(seq_len, np_backend.device, str(np_backend.dtype))
