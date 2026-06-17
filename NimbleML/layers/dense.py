"""Dense (fully connected) layer"""
from math import prod, sqrt
from random import uniform

from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


def fused_dense(x, weights, bias):
    """Fused matmul + bias with one autograd node."""
    in_shape = x.shape
    in_features = in_shape[-1]
    out_features = weights.shape[1]
    row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1

    x_arr = Tensor._asarray(x.data).reshape(row_count, in_features)
    w_arr = Tensor._asarray(weights.data).reshape(in_features, out_features)
    out2d = x_arr @ w_arr
    if bias is not None:
        out2d = out2d + Tensor._asarray(bias.data).reshape(out_features)

    save_x = _save_for_backward(x_arr)

    out_shape = in_shape[:-1] + (out_features,)
    out_arr = out2d.reshape(out_shape)

    requires_grad = x.requires_grad or weights.requires_grad or (bias is not None and bias.requires_grad)
    children = [t for t in (x, weights, bias) if t is not None]
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=requires_grad,
        _children=tuple(children),
        _op="fused_dense",
    )

    def _backward():
        if out.grad is None:
            return

        grad_out = Tensor._asarray(out.grad).reshape(row_count, out_features)
        w_arr = Tensor._asarray(weights.data).reshape(in_features, out_features)
        if weights.requires_grad:
            grad_w = save_x.T @ grad_out
            weights._accumulate_grad(grad_w.ravel())
        if bias is not None and bias.requires_grad:
            bias._accumulate_grad(np.sum(grad_out, axis=0).ravel())
        if x.requires_grad:
            grad_x = grad_out @ w_arr.T
            x._accumulate_grad(grad_x.reshape(in_shape).ravel())

    out._backward = _backward
    return out


class Dense(Module):
    """Public class Dense."""
    def __init__(self, in_features, out_features, bias=True, weight_scale=None):
        self.in_features = in_features
        self.out_features = out_features
        scale = weight_scale if weight_scale is not None else 1.0 / sqrt(max(1, in_features))
        self.weights = Tensor(
            [uniform(-1, 1) * scale for _ in range(in_features * out_features)],
            (in_features, out_features),
            requires_grad=True,
        )
        self.biases = (Tensor([0.0] * out_features, (out_features,), requires_grad=True) if bias else None)

    def forward(self, inputs):
        """Public function forward."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {inputs.shape[-1]}")
        return fused_dense(inputs, self.weights, self.biases)

    def parameters(self):
        """Public function parameters."""
        params = [self.weights]
        if self.biases is not None:
            params.append(self.biases)
        return params
