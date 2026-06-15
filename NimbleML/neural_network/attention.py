# attention.py
# Scaled dot-product attention (single-head)
from NimbleML.layers import Dense
from NimbleML.activations import Softmax
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def make_causal_mask(seq_len):
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

def _swap_last_two(tensor):
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("swap_last_two needs at least 2 dimensions.")

    arr = Tensor._asarray(tensor.data).reshape(shape)
    out_arr = np.swapaxes(arr, -2, -1)
    out_shape = shape[:-2] + (shape[-1], shape[-2])
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="swap_last_two",
    )

    def _backward():
        if out.grad is None or not tensor.requires_grad:
            return
        grad_out = out.grad.reshape(out_shape)
        grad_in = np.swapaxes(grad_out, -2, -1)
        tensor._accumulate_grad(grad_in.ravel())

    out._backward = _backward
    return out


def _split_heads(tensor, batch, seq_len, num_heads, d_k):
    """(batch, seq, d_model) -> (batch * num_heads, seq, d_k)."""
    shape_4d = (batch, seq_len, num_heads, d_k)
    t = tensor.reshape(shape_4d)
    arr = Tensor._asarray(t.data).reshape(shape_4d)
    out_arr = np.transpose(arr, (0, 2, 1, 3))
    out_shape = (batch * num_heads, seq_len, d_k)
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(t,),
        _op="split_heads",
    )

    def _backward():
        if out.grad is None or not tensor.requires_grad:
            return
        grad_out = out.grad.reshape(batch, num_heads, seq_len, d_k)
        grad_in = np.transpose(grad_out, (0, 2, 1, 3))
        t._accumulate_grad(grad_in.ravel())

    out._backward = _backward
    return out


def _merge_heads(tensor, batch, seq_len, num_heads, d_k):
    """(batch * num_heads, seq, d_k) -> (batch, seq, d_model)."""
    shape_4d = (batch, num_heads, seq_len, d_k)
    t = tensor.reshape(shape_4d)
    arr = Tensor._asarray(t.data).reshape(shape_4d)
    out_arr = np.transpose(arr, (0, 2, 1, 3))
    out_shape = (batch, seq_len, num_heads * d_k)
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(t,),
        _op="merge_heads",
    )

    def _backward():
        if out.grad is None or not tensor.requires_grad:
            return
        grad_out = out.grad.reshape(batch, seq_len, num_heads, d_k)
        grad_in = np.transpose(grad_out, (0, 2, 1, 3))
        t._accumulate_grad(grad_in.ravel())

    out._backward = _backward
    return out


class Attention(Module):
    def __init__(self, d_k):
        self.d_k = d_k
        self.scale = float(d_k) ** 0.5
        self.softmax = Softmax(axis=-1)

    def forward(self, Q, K, V, mask=None):
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

        batch, seq_len, _ = Q.shape
        scores = Q @ _swap_last_two(K)
        scores = scores / self.scale

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float64)
            if mask_arr.shape != (seq_len, seq_len):
                raise ValueError(f"mask must be ({seq_len}, {seq_len}), got {mask_arr.shape}")
            mask_tensor = Tensor(mask_arr.ravel(), mask_arr.shape, requires_grad=False)
            scores = scores + mask_tensor

        weights = self.softmax(scores)
        return weights @ V

    def parameters(self):
        return []


class MultiHeadAttention(Module):
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
        self.attention = Attention(d_k=self.d_k)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model {self.d_model}, got {d_model}")

        Q = _split_heads(self.W_q(x), batch, seq_len, self.num_heads, self.d_k)
        K = _split_heads(self.W_k(x), batch, seq_len, self.num_heads, self.d_k)
        V = _split_heads(self.W_v(x), batch, seq_len, self.num_heads, self.d_k)

        out = self.attention.forward(Q, K, V, mask=mask)
        out = _merge_heads(out, batch, seq_len, self.num_heads, self.d_k)
        return self.W_o(out)

    def parameters(self):
        params = []
        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            params.extend(layer.parameters())
        return params