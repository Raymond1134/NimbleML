"""Scaled dot-product attention (single-head and multi-head)"""
from functools import lru_cache
from NimbleML.activations.softmax import softmax_backward, softmax_forward
from NimbleML.layers import Dense
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


@lru_cache(maxsize=None)
def make_causal_mask(seq_len):
    """Public function make_causal_mask."""
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)


@lru_cache(maxsize=None)
def causal_mask_tensor(seq_len):
    """Cached (seq_len, seq_len) additive causal mask as a no-grad Tensor."""
    mask_arr = np.asarray(make_causal_mask(seq_len), dtype=np_backend.dtype)
    return Tensor(mask_arr.ravel(), mask_arr.shape, requires_grad=False)


def _resolve_mask(mask, seq_len):
    if mask is None:
        return None
    if isinstance(mask, Tensor):
        mask_tensor = mask
        if mask_tensor.shape != (seq_len, seq_len):
            raise ValueError(f"mask must be ({seq_len}, {seq_len}), got {mask_tensor.shape}")
        return Tensor._asarray(mask_tensor.data).reshape(mask_tensor.shape)
    mask_arr = np.asarray(mask, dtype=np_backend.dtype)
    if mask_arr.shape != (seq_len, seq_len):
        raise ValueError(f"mask must be ({seq_len}, {seq_len}), got {mask_arr.shape}")
    return mask_arr


def _split_heads(tensor, batch, seq_len, num_heads, d_k):
    """(batch, seq, d_model) -> (batch * num_heads, seq, d_k)."""
    expected = (batch, seq_len, num_heads * d_k)
    if tensor.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {tensor.shape}")

    shape_4d = (batch, seq_len, num_heads, d_k)
    arr = Tensor._asarray(tensor.data).reshape(shape_4d)
    out_arr = np.transpose(arr, (0, 2, 1, 3))
    out_shape = (batch * num_heads, seq_len, d_k)
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="split_heads",
    )

    def _backward():
        if out.grad is None or not tensor.requires_grad:
            return
        grad_out = out.grad.reshape(batch, num_heads, seq_len, d_k)
        grad_in = np.transpose(grad_out, (0, 2, 1, 3))
        tensor._accumulate_grad(grad_in.ravel())

    out._backward = _backward
    return out


def _merge_heads(tensor, batch, seq_len, num_heads, d_k):
    """(batch * num_heads, seq, d_k) -> (batch, seq, d_model)."""
    expected = (batch * num_heads, seq_len, d_k)
    if tensor.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {tensor.shape}")

    shape_4d = (batch, num_heads, seq_len, d_k)
    arr = Tensor._asarray(tensor.data).reshape(shape_4d)
    out_arr = np.transpose(arr, (0, 2, 1, 3))
    out_shape = (batch, seq_len, num_heads * d_k)
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="merge_heads",
    )

    def _backward():
        if out.grad is None or not tensor.requires_grad:
            return
        grad_out = out.grad.reshape(batch, seq_len, num_heads, d_k)
        grad_in = np.transpose(grad_out, (0, 2, 1, 3))
        tensor._accumulate_grad(grad_in.ravel())

    out._backward = _backward
    return out


def scaled_dot_product_attention(Q, K, V, scale, mask=None):
    """
    Fused attention: QK^T / scale, optional mask, softmax, @V.

    Q, K, V: (batch, seq, d_k).
    mask: optional (seq, seq) additive mask (Tensor or array).
    """
    q_shape = Q.shape
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(
            f"Expected Q/K/V shapes to match, got {Q.shape}, {K.shape}, {V.shape}"
        )
    if Q.ndim != 3:
        raise ValueError(f"Expected 3D Q/K/V, got ndim={Q.ndim}")

    seq_len = q_shape[-2]
    mask_arr = _resolve_mask(mask, seq_len)

    q_arr = Tensor._asarray(Q.data).reshape(q_shape)
    k_arr = Tensor._asarray(K.data).reshape(q_shape)
    v_arr = Tensor._asarray(V.data).reshape(q_shape)

    scores = np.matmul(q_arr, np.swapaxes(k_arr, -2, -1)) / scale
    if mask_arr is not None:
        scores = scores + mask_arr
    probs = softmax_forward(scores, axis=-1)
    out_arr = np.matmul(probs, v_arr)

    save_q = _save_for_backward(q_arr)
    save_k = _save_for_backward(k_arr)
    save_v = _save_for_backward(v_arr)

    requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad
    out = Tensor(
        out_arr.ravel(),
        out_arr.shape,
        requires_grad=requires_grad,
        _children=(Q, K, V),
        _op="scaled_dot_product_attention",
    )

    def _backward():
        if out.grad is None:
            return

        grad_out = Tensor._asarray(out.grad).reshape(out_arr.shape)
        scale_f = float(scale)
        scores = np.matmul(save_q, np.swapaxes(save_k, -2, -1)) / scale_f
        if mask_arr is not None:
            scores = scores + mask_arr
        probs = softmax_forward(scores, axis=-1)
        grad_probs = np.matmul(grad_out, np.swapaxes(save_v, -2, -1))
        grad_scores = softmax_backward(grad_probs, probs, axis=-1)
        grad_scores = grad_scores / scale_f

        if Q.requires_grad:
            grad_q = np.matmul(grad_scores, save_k)
            Q._accumulate_grad(grad_q.ravel())
        if K.requires_grad:
            grad_k = np.matmul(np.swapaxes(grad_scores, -2, -1), save_q)
            K._accumulate_grad(grad_k.ravel())
        if V.requires_grad:
            grad_v = np.matmul(np.swapaxes(probs, -2, -1), grad_out)
            V._accumulate_grad(grad_v.ravel())

    out._backward = _backward
    return out


class Attention(Module):
    """Public class Attention."""
    def __init__(self, d_k):
        self.d_k = d_k
        self.scale = float(d_k) ** 0.5

    def forward(self, Q, K, V, mask=None):
        """Public function forward."""
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError(
                f"Expected 3 dimensions, got {Q.ndim} for Q, {K.ndim} for K, {V.ndim} for V"
            )
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError(
                f"Expected shapes to match, got {Q.shape} for Q, {K.shape} for K, {V.shape} for V"
            )
        if Q.shape[-1] != self.d_k:
            raise ValueError(f"Expected last dim {self.d_k}, got {Q.shape[-1]}")

        return scaled_dot_product_attention(Q, K, V, self.scale, mask=mask)

    def parameters(self):
        """Public function parameters."""
        return []


class MultiHeadAttention(Module):
    """Public class MultiHeadAttention."""
    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = Dense(d_model, d_model)
        self.W_k = Dense(d_model, d_model)
        self.W_v = Dense(d_model, d_model)
        self.W_o = Dense(d_model, d_model)
        self.scale = float(self.d_k) ** 0.5

    def forward(self, x, mask=None):
        """Public function forward."""
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model {self.d_model}, got {d_model}")

        Q = _split_heads(self.W_q(x), batch, seq_len, self.num_heads, self.d_k)
        K = _split_heads(self.W_k(x), batch, seq_len, self.num_heads, self.d_k)
        V = _split_heads(self.W_v(x), batch, seq_len, self.num_heads, self.d_k)

        out = scaled_dot_product_attention(Q, K, V, self.scale, mask=mask)
        out = _merge_heads(out, batch, seq_len, self.num_heads, self.d_k)
        return self.W_o(out)

    def parameters(self):
        """Public function parameters."""
        params = []
        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            params.extend(layer.parameters())
        return params
