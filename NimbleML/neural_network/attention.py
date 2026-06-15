# attention.py
# Scaled dot-product attention (single-head)
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
