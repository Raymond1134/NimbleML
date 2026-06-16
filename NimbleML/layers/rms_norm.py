"""Root mean square layer normalization over the last dimension."""
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class RMSNorm(Module):
    """RMSNorm: scale normalized activations by RMS (no mean centering or bias)."""

    def __init__(self, normalized_shape, epsilon=1e-5):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.gamma = Tensor(
            np.ones(normalized_shape),
            (normalized_shape,),
            requires_grad=True,
        )

    def forward(self, inputs):
        """Normalize by RMS over the last dimension and apply ``gamma``."""
        if inputs.shape[-1] != self.normalized_shape:
            raise ValueError(f"Expected last dim {self.normalized_shape}, got {inputs.shape[-1]}")

        ms = (inputs ** 2).mean(axis=-1, keepdims=True)
        rms = (ms + self.epsilon).sqrt()
        return inputs / rms * self.gamma

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma]
