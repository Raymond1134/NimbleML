"""Transformer FFN: expand -> GELU -> project back (per-token MLP)"""
from math import prod

from NimbleML.activations.gelu import gelu_backward, gelu_forward
from NimbleML.layers import Dense
from NimbleML.neural_network.module import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


def fused_feed_forward(x, weights1, bias1, weights2, bias2):
    """
    Fused FFN: matmul+bias -> GELU -> matmul+bias with one autograd node.

    Uses a flattened (N, d) layout so each matmul is a single cuBLAS GEMM.
    """
    in_shape = x.shape
    if in_shape[-1] != weights1.shape[0]:
        raise ValueError(
            f"Expected last dim {weights1.shape[0]}, got {in_shape[-1]}"
        )
    if weights1.shape[1] != weights2.shape[0]:
        raise ValueError(
            f"Dense1 out ({weights1.shape[1]}) must match Dense2 in ({weights2.shape[0]})"
        )
    if weights2.shape[1] != in_shape[-1]:
        raise ValueError(
            f"Expected output last dim {in_shape[-1]}, got {weights2.shape[1]}"
        )

    row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1
    d_in = in_shape[-1]
    d_hidden = weights1.shape[1]
    d_out = weights2.shape[1]

    x_arr = Tensor._asarray(x.data).reshape(row_count, d_in)
    w1 = Tensor._asarray(weights1.data).reshape(d_in, d_hidden)
    w2 = Tensor._asarray(weights2.data).reshape(d_hidden, d_out)
    b1 = Tensor._asarray(bias1.data).reshape(d_hidden) if bias1 is not None else None
    b2 = Tensor._asarray(bias2.data).reshape(d_out) if bias2 is not None else None

    pre_act = np.matmul(x_arr, w1)
    if b1 is not None:
        pre_act = pre_act + b1
    hidden, _tanh_u = gelu_forward(pre_act)
    out2d = np.matmul(hidden, w2)
    if b2 is not None:
        out2d = out2d + b2

    save_x = _save_for_backward(x_arr)
    save_pre = _save_for_backward(pre_act)

    out_shape = in_shape[:-1] + (d_out,)
    out_arr = out2d.reshape(out_shape)

    requires_grad = any(
        t is not None and t.requires_grad
        for t in (x, weights1, bias1, weights2, bias2)
    )
    children = tuple(t for t in (x, weights1, bias1, weights2, bias2) if t is not None)
    out = Tensor(
        out_arr.ravel(),
        out_shape,
        requires_grad=requires_grad,
        _children=children,
        _op="fused_feed_forward",
    )

    def _backward():
        if out.grad is None:
            return

        grad_out = Tensor._asarray(out.grad).reshape(row_count, d_out)

        w1 = Tensor._asarray(weights1.data).reshape(d_in, d_hidden)
        w2 = Tensor._asarray(weights2.data).reshape(d_hidden, d_out)

        if weights2.requires_grad:
            hidden, _ = gelu_forward(save_pre)
            grad_w2 = np.matmul(hidden.T, grad_out)
            weights2._accumulate_grad(grad_w2.ravel())
        if bias2 is not None and bias2.requires_grad:
            bias2._accumulate_grad(np.sum(grad_out, axis=0).ravel())

        grad_hidden = np.matmul(grad_out, w2.T)
        grad_pre_act = gelu_backward(grad_hidden, save_pre)

        if weights1.requires_grad:
            grad_w1 = np.matmul(save_x.T, grad_pre_act)
            weights1._accumulate_grad(grad_w1.ravel())
        if bias1 is not None and bias1.requires_grad:
            bias1._accumulate_grad(np.sum(grad_pre_act, axis=0).ravel())
        if x.requires_grad:
            grad_x = np.matmul(grad_pre_act, w1.T)
            x._accumulate_grad(grad_x.reshape(in_shape).ravel())

    out._backward = _backward
    return out


class FeedForward(Module):
    """Public class FeedForward."""
    def __init__(self, d_model, ff_mult=4):
        hidden = ff_mult * d_model
        self.dense1 = Dense(d_model, hidden)
        self.dense2 = Dense(hidden, d_model)

    def forward(self, x):
        """Public function forward."""
        return fused_feed_forward(
            x,
            self.dense1.weights,
            self.dense1.biases,
            self.dense2.weights,
            self.dense2.biases,
        )

    def parameters(self):
        """Public function parameters."""
        params = []
        for layer in (self.dense1, self.dense2):
            params.extend(layer.parameters())
        return params
