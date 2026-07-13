"""Scaled dot-product attention."""
from NimbleML.kernels.fused_sdpa import fused_sdpa_backward, fused_sdpa_forward
from NimbleML.layers import Dense
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


class Attention(Module):
    """Scaled dot-product attention (single-head)."""
    def __init__(self, d_k):
        self.d_k = d_k
        self.scale = float(d_k) ** 0.5

    def forward(self, Q, K, V, mask=None):
        """Applies scaled dot-product attention.

        Args:
            Q (Tensor): Query tensor (batch, seq, d_k).
            K (Tensor): Key tensor (batch, seq, d_k).
            V (Tensor): Value tensor (batch, seq, d_k).
            mask (array-like or Tensor, optional): Attention mask.

        Returns:
            Tensor: Output tensor (batch, seq, d_k).
        
        Raises:
            ValueError:
                - If Q, K, V are not 3D.
                - If Q, K, V shapes do not match.
                - If Q.shape[-1] != d_k.
        
        Examples:
            >>> attention = Attention(d_k=128)
            >>> Q = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3, 128), requires_grad=True)
            >>> K = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3, 128), requires_grad=True)
            >>> V = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3, 128), requires_grad=True)
            >>> mask = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), shape=(2, 3), requires_grad=True)
            >>> output = attention(Q, K, V, mask)
        """
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

        return self._scaled_dot_product_attention(Q, K, V, self.scale, mask=mask)

    def parameters(self):
        """Returns learnable parameters (none for this module)."""
        return []

    @staticmethod
    def _scaled_dot_product_attention(Q, K, V, scale, mask=None):
        q_shape = Q.shape
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError(
                f"Expected Q/K/V shapes to match, got {Q.shape}, {K.shape}, {V.shape}"
            )
        if Q.ndim != 3:
            raise ValueError(f"Expected 3D Q/K/V, got ndim={Q.ndim}")

        seq_len = q_shape[-2]
        if mask is None:
            mask_arr = None
        elif isinstance(mask, Tensor):
            mask_tensor = mask
            if mask_tensor.shape != (seq_len, seq_len):
                raise ValueError(f"mask must be ({seq_len}, {seq_len}), got {mask_tensor.shape}")
            mask_arr = mask_tensor._view()
        else:
            mask_arr = np.asarray(mask, dtype=np_backend.dtype)
            if mask_arr.shape != (seq_len, seq_len):
                raise ValueError(f"mask must be ({seq_len}, {seq_len}), got {mask_arr.shape}")

        q_arr = Q._view(q_shape)
        k_arr = K._view(q_shape)
        v_arr = V._view(q_shape)

        out_arr, probs = fused_sdpa_forward(q_arr, k_arr, v_arr, scale, mask_arr)

        save_q = _save_for_backward(q_arr)
        save_k = _save_for_backward(k_arr)
        save_v = _save_for_backward(v_arr)
        # Flash meta is a small dict (m, l); dense path stores full probs.
        save_probs = probs if isinstance(probs, dict) else _save_for_backward(probs)

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

            grad_out = _grad_out(out, out_arr.shape)
            grad_q, grad_k, grad_v = fused_sdpa_backward(
                grad_out, save_q, save_k, save_v, save_probs, scale
            )

            if Q.requires_grad:
                Q._accumulate_grad(grad_q.ravel())
            if K.requires_grad:
                K._accumulate_grad(grad_k.ravel())
            if V.requires_grad:
                V._accumulate_grad(grad_v.ravel())

        out._backward = _backward
        return out


class MultiHeadAttention(Module):
    """Multi-head scaled dot-product attention."""
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
        # (cos, sin) RoPE cache, set by GPT when use_rope=True (fused path only).
        self._rope_cache = None

    def forward(self, x, mask=None):
        """Applies multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch, seq, d_model).
            mask (array-like or Tensor, optional): Attention mask.

        Returns:
            Tensor: Output tensor of shape (batch, seq, d_model).
        
        Raises:
            ValueError: If d_model does not match the expected value.
        
        Examples:
            >>> attention = MultiHeadAttention(d_model=768, num_heads=12)
            >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3, 768), requires_grad=True)
            >>> mask = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), shape=(2, 3), requires_grad=True)
            >>> output = attention(x, mask)
        """
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model {self.d_model}, got {d_model}")

        Q = self._split_heads(self.W_q(x), batch, seq_len, self.num_heads, self.d_k)
        K = self._split_heads(self.W_k(x), batch, seq_len, self.num_heads, self.d_k)
        V = self._split_heads(self.W_v(x), batch, seq_len, self.num_heads, self.d_k)

        out = Attention._scaled_dot_product_attention(Q, K, V, self.scale, mask=mask)
        out = self._merge_heads(out, batch, seq_len, self.num_heads, self.d_k)
        return self.W_o(out)

    def parameters(self):
        """Returns all learnable parameters in projection layers."""
        params = []
        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            params.extend(layer.parameters())
        return params

    @staticmethod
    def _split_heads(tensor, batch, seq_len, num_heads, d_k):
        expected = (batch, seq_len, num_heads * d_k)
        if tensor.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {tensor.shape}")

        shape_4d = (batch, seq_len, num_heads, d_k)
        arr = tensor._view(shape_4d)
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
            grad_out = _grad_out(out, (batch, num_heads, seq_len, d_k))
            grad_in = np.transpose(grad_out, (0, 2, 1, 3))
            tensor._accumulate_grad(grad_in.ravel())

        out._backward = _backward
        return out

    @staticmethod
    def _merge_heads(tensor, batch, seq_len, num_heads, d_k):
        expected = (batch * num_heads, seq_len, d_k)
        if tensor.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {tensor.shape}")

        shape_4d = (batch, num_heads, seq_len, d_k)
        arr = tensor._view(shape_4d)
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
            grad_out = _grad_out(out, (batch, seq_len, num_heads, d_k))
            grad_in = np.transpose(grad_out, (0, 2, 1, 3))
            tensor._accumulate_grad(grad_in.ravel())

        out._backward = _backward
        return out
