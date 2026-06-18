"""Package exports and public API surface."""

from .axis import normalize_axis, normalize_axes
from .shape import kernel_dims
from .clip_grad import clip_grad_norm_
from .np_backend import as_int64, apply_runtime_config, device, get_dtype, np, set_device, set_dtype, using_gpu
from .tensor import Tensor

__all__ = [
    "Tensor",
    "np",
    "device",
    "using_gpu",
    "set_device",
    "set_dtype",
    "get_dtype",
    "apply_runtime_config",
    "as_int64",
    "clip_grad_norm_",
    "normalize_axis",
    "normalize_axes",
    "kernel_dims",
]
