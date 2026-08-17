"""Dense (fully connected) layer."""
from math import prod, sqrt
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


class Dense(Module):
    """Linear map on the last axis: ``y = x @ W + b`` with ``W.shape = (in, out)``."""

    def __init__(self, in_features, out_features, bias=True, weight_scale=None):
        self.in_features = in_features
        self.out_features = out_features
        scale = weight_scale if weight_scale is not None else 1.0 / sqrt(max(1, in_features))
        self.weights = Tensor(
            np.random.uniform(-scale, scale, size=(in_features, out_features)),
            (in_features, out_features),
            requires_grad=True,
        )
        self.biases = (
            Tensor(np.zeros(out_features), (out_features,), requires_grad=True) if bias else None
        )

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {inputs.shape[-1]}")

        weights = self.weights
        bias = self.biases
        in_shape = inputs.shape
        in_features = self.in_features
        out_features = self.out_features
        row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1

        x_arr = inputs._view((row_count, in_features))
        w_arr = weights._view((in_features, out_features))
        w_T = np.swapaxes(w_arr, -2, -1)
        out2d = x_arr @ w_arr
        if bias is not None:
            out2d = out2d + bias._view(out_features)

        save_x = _save_for_backward(x_arr)

        out_shape = in_shape[:-1] + (out_features,)
        out_arr = out2d.reshape(out_shape)

        requires_grad = (
            inputs.requires_grad
            or weights.requires_grad
            or (bias is not None and bias.requires_grad)
        )
        children = tuple(t for t in (inputs, weights, bias) if t is not None)
        out = Tensor(
            out_arr.ravel(),
            out_shape,
            requires_grad=requires_grad,
            _children=children,
            _op="dense",
        )

        def _backward():
            if out.grad is None:
                return
            grad_out = _grad_out(out, (row_count, out_features))
            if weights.requires_grad:
                grad_w = np.matmul(np.swapaxes(save_x, -2, -1), grad_out)
                weights._accumulate_grad(grad_w.ravel())
            if bias is not None and bias.requires_grad:
                bias._accumulate_grad(np.sum(grad_out, axis=0).ravel())
            if inputs.requires_grad:
                grad_x = np.matmul(grad_out, w_T)
                inputs._accumulate_grad(grad_x.reshape(in_shape).ravel())

        out._backward = _backward
        return out

    def parameters(self):
        params = [self.weights]
        if self.biases is not None:
            params.append(self.biases)
        return params
