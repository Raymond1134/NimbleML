"""Transformer feedforward."""
from math import prod
from NimbleML.activations.gelu import gelu_backward, gelu_forward
from NimbleML.layers import Dense
from NimbleML.neural_network.module import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


class FeedForward(Module):
    """Transformer feedforward network (per-token MLP)."""

    def __init__(self, d_model, ff_mult=4):
        hidden = ff_mult * d_model
        self.dense1 = Dense(d_model, hidden)
        self.dense2 = Dense(hidden, d_model)

    def forward(self, x):
        """Applies the Transformer feedforward network.

        This module processes each token independently using a 2-layer MLP:

            1. Linear projection: d_model → ff_mult * d_model
            2. GELU activation
            3. Linear projection: back to d_model

        Args:
            x (Tensor): Input tensor of shape (batch, seq, d_model).

        Returns:
            Tensor: Output tensor of shape (batch, seq, d_model).
        
        Raises:
            ValueError:
                - If the input tensor does not match the expected shape.
                - If the dense layers do not match the expected shapes.
                - If the output tensor does not match the expected shape.
        
        Examples:
            >>> feed_forward = FeedForward(d_model=768, ff_mult=4)
            >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3, 768), requires_grad=True)
            >>> output = feed_forward(x)
        """
        weights1 = self.dense1.weights
        bias1 = self.dense1.biases
        weights2 = self.dense2.weights
        bias2 = self.dense2.biases

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
        hidden, tanh_u = gelu_forward(pre_act)
        out2d = np.matmul(hidden, w2)
        if b2 is not None:
            out2d = out2d + b2

        save_x = _save_for_backward(x_arr)
        save_pre = _save_for_backward(pre_act)
        save_tanh_u = _save_for_backward(tanh_u)
        save_hidden = _save_for_backward(hidden)

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

            grad_out = _grad_out(out, (row_count, d_out))
            w1 = Tensor._asarray(weights1.data).reshape(d_in, d_hidden)
            w2 = Tensor._asarray(weights2.data).reshape(d_hidden, d_out)
            w2_T = np.ascontiguousarray(np.swapaxes(w2, -2, -1))
            w1_T = np.ascontiguousarray(np.swapaxes(w1, -2, -1))

            if weights2.requires_grad:
                grad_w2 = np.matmul(np.ascontiguousarray(np.swapaxes(save_hidden, -2, -1)), grad_out)
                weights2._accumulate_grad(grad_w2.ravel())
            if bias2 is not None and bias2.requires_grad:
                bias2._accumulate_grad(np.sum(grad_out, axis=0).ravel())

            grad_hidden = np.matmul(grad_out, w2_T)
            grad_pre_act = gelu_backward(grad_hidden, save_pre, save_tanh_u)

            if weights1.requires_grad:
                grad_w1 = np.matmul(np.ascontiguousarray(np.swapaxes(save_x, -2, -1)), grad_pre_act)
                weights1._accumulate_grad(grad_w1.ravel())
            if bias1 is not None and bias1.requires_grad:
                bias1._accumulate_grad(np.sum(grad_pre_act, axis=0).ravel())
            if x.requires_grad:
                grad_x = np.matmul(grad_pre_act, w1_T)
                x._accumulate_grad(grad_x.reshape(in_shape).ravel())

        out._backward = _backward
        return out

    def parameters(self):
        """Returns all learnable parameters in the feedforward network."""
        params = []
        for layer in (self.dense1, self.dense2):
            params.extend(layer.parameters())
        return params
