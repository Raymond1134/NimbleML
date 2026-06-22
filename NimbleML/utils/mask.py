"""Attention mask helpers."""
from functools import lru_cache

from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


@lru_cache(maxsize=None)
def make_causal_mask(seq_len):
    """Upper-triangular (seq_len, seq_len) additive mask for causal attention."""
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)


@lru_cache(maxsize=None)
def causal_mask_tensor(seq_len):
    """Cached (seq_len, seq_len) additive causal mask as a no-grad Tensor."""
    mask_arr = np.asarray(make_causal_mask(seq_len), dtype=np_backend.dtype)
    return Tensor(mask_arr.ravel(), mask_arr.shape, requires_grad=False)
