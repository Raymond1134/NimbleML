"""Package exports and public API surface."""

from .np_backend import device, np, set_device, using_gpu
from .tensor import Tensor

__all__ = ["Tensor", "np", "device", "using_gpu", "set_device"]
