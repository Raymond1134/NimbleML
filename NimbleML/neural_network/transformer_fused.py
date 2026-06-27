"""Fused pre-norm transformer blocks."""
from __future__ import annotations
from NimbleML.layers import RMSNorm
from NimbleML.neural_network._fused_arrays import (
    fused_block_backward_arrays,
    fused_block_forward_arrays,
    fused_trunk_backward_arrays,
    fused_trunk_forward_arrays,
)
from NimbleML.neural_network.attention import MultiHeadAttention
from NimbleML.neural_network.feed_forward import FeedForward
from NimbleML.neural_network.module import Module
from NimbleML.utils.mask import causal_mask_tensor
from NimbleML.utils.tensor import Tensor, _grad_out


def _mask_to_array(mask, seq_len):
    from NimbleML.utils import np_backend
    from NimbleML.utils.np_backend import np

    if mask is None:
        return None
    if isinstance(mask, Tensor):
        mask_arr = Tensor._asarray(mask.data).reshape(mask.shape)
    else:
        mask_arr = np.asarray(mask, dtype=np_backend.dtype)
    if mask_arr.shape != (seq_len, seq_len):
        raise ValueError(f"mask must be ({seq_len}, {seq_len}), got {mask_arr.shape}")
    return mask_arr


class FusedTransformerBlock(Module):
    """Pre-norm block with a single autograd closure."""

    def __init__(self, d_model, num_heads, ff_mult=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult=ff_mult)

    def forward(self, x, mask=None):
        if mask is None:
            mask = causal_mask_tensor(x.shape[1])

        mask_arr = _mask_to_array(mask, x.shape[1])
        x_arr = Tensor._asarray(x.data).reshape(x.shape)
        out_arr, ctx = fused_block_forward_arrays(x_arr, self, mask_arr)

        out = Tensor(
            out_arr.ravel(),
            out_arr.shape,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="fused_transformer_block",
        )

        def _backward():
            if out.grad is None:
                return
            grad_out = _grad_out(out, x.shape)
            grad_x = fused_block_backward_arrays(grad_out, ctx)
            if x.requires_grad:
                x._accumulate_grad(grad_x.ravel())

        out._backward = _backward
        return out

    def parameters(self):
        params = []
        for layer in (self.ln1, self.mha, self.ln2, self.ffn):
            params.extend(layer.parameters())
        return params


class FusedGPTTrunk(Module):
    """All transformer blocks + final RMSNorm in one autograd node."""

    def __init__(self, blocks, ln: RMSNorm):
        self.blocks = list(blocks)
        self.ln = ln

    def forward(self, x, mask=None):
        if mask is None:
            mask = causal_mask_tensor(x.shape[1])

        mask_arr = _mask_to_array(mask, x.shape[1])
        x_arr = Tensor._asarray(x.data).reshape(x.shape)
        out_arr, trunk_ctx = fused_trunk_forward_arrays(x_arr, self.blocks, self.ln, mask_arr)

        out = Tensor(
            out_arr.ravel(),
            out_arr.shape,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="fused_gpt_trunk",
        )

        def _backward():
            if out.grad is None:
                return
            grad_out = _grad_out(out, x.shape)
            grad_x = fused_trunk_backward_arrays(grad_out, trunk_ctx)
            if x.requires_grad:
                x._accumulate_grad(grad_x.ravel())

        out._backward = _backward
        return out

    def parameters(self):
        params = []
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln.parameters())
        return params
