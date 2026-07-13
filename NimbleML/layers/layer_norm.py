"""Layer normalization over the last dimension."""
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class LayerNorm(Module):
    """Layer normalization module.

    Normalizes activations across the last dimension of the input tensor,
    then applies learnable scale and shift parameters:

        y = gamma * ((x - mean) / sqrt(var + epsilon)) + beta

    where the mean and variance are computed independently for each
    sample over the last dimension.
    """
    def __init__(self, normalized_shape, epsilon=1e-5):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.gamma = Tensor(
            np.ones(normalized_shape),
            (normalized_shape,),
            requires_grad=True,
        )
        self.beta = Tensor(
            np.zeros(normalized_shape),
            (normalized_shape,),
            requires_grad=True,
        )

    def forward(self, inputs):
        """Applies layer normalization to the input tensor.

        Args:
            inputs (Tensor): Input tensor whose last dimension matches ``normalized_shape``.

        Returns:
            Tensor: Normalized tensor with the same shape as ``inputs``.

        Raises:
            ValueError: If the last input dimension does not match ``normalized_shape``.
        
        Examples:
            >>> layer = LayerNorm(normalized_shape=10)
            >>> inputs = Tensor(np.random.randn(10, 10), (10, 10))
            >>> output = layer.forward(inputs)
        """
        if inputs.shape[-1] != self.normalized_shape:
            raise ValueError(f"Expected last dim {self.normalized_shape}, got {inputs.shape[-1]}")
        
        mean = inputs.mean(axis=-1, keepdims=True)
        centered = inputs - mean
        variance = (centered ** 2).mean(axis=-1, keepdims=True)
        std = (variance + self.epsilon).sqrt()
        normalized = centered / std
        return normalized * self.gamma + self.beta

    def parameters(self):
        """Returns learnable parameters of the layer.

        Returns:
            list[Tensor]: Scale parameter ``gamma`` and shift parameter ``beta``.
        
        Examples:
            >>> layer = LayerNorm(normalized_shape=10)
            >>> params = layer.parameters()
        """
        return [self.gamma, self.beta]
