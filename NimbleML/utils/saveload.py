"""Save and load model parameters (.npz)"""
import numpy as host_np

from NimbleML.neural_network.module import Module, Sequential
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def named_parameters(module, prefix=""):
    """Yield (name, Tensor) for every learnable parameter in the module tree."""
    if isinstance(module, Sequential):
        for i, layer in enumerate(module.layers):
            child_prefix = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
            yield from named_parameters(layer, child_prefix)
        return

    if not isinstance(module, Module):
        return

    for name, value in vars(module).items():
        if name.startswith("_"):
            continue

        child_prefix = f"{prefix}.{name}" if prefix else name

        if isinstance(value, Tensor):
            if value.requires_grad:
                yield child_prefix, value
        elif isinstance(value, Module):
            yield from named_parameters(value, child_prefix)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, Module):
                    yield from named_parameters(item, f"{child_prefix}.{i}")


def _to_host_array(param):
    arr = param.data.reshape(param.shape)
    get = getattr(arr, "get", None)
    if get is not None:
        arr = get()
    return host_np.asarray(arr, dtype=np_backend.dtype)


def save(model, path):
    """Save all learnable parameters to a .npz archive."""
    state = {name: _to_host_array(param) for name, param in named_parameters(model)}
    if not state:
        raise ValueError("Model has no parameters to save.")
    host_np.savez(path, **state)


def load(model, path):
    """Load parameters from .npz into an existing model (same architecture)."""
    expected = dict(named_parameters(model))
    if not expected:
        raise ValueError("Model has no parameters to load.")

    with host_np.load(path) as data:
        names = set(data.files)
        expected_names = set(expected)
        if names != expected_names:
            missing = sorted(expected_names - names)
            extra = sorted(names - expected_names)
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"unexpected {extra}")
            raise ValueError(f"Checkpoint mismatch: {', '.join(parts)}")

        for name, param in expected.items():
            loaded = host_np.asarray(data[name], dtype=param.data.dtype).reshape(param.shape)
            if loaded.size != param.size:
                raise ValueError(f"Shape mismatch for {name}: expected {param.shape}, got {loaded.shape}")
            param.data[...] = np.asarray(loaded.ravel(), dtype=param.data.dtype)

    return model
