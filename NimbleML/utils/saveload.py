"""Save and load model parameters and training checkpoints (.npz)."""
from __future__ import annotations
import json
from typing import Any
import numpy as host_np
from NimbleML.neural_network.module import Module, Sequential
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor

_CHECKPOINT_VERSION = 1
_META_KEY = "_checkpoint_meta"
_MODEL_PREFIX = "model/"


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


def _to_host_ndarray(arr):
    get = getattr(arr, "get", None)
    if get is not None:
        arr = get()
    return host_np.asarray(arr)


def _load_ndarray_into(target, loaded):
    arr = np.asarray(loaded.ravel(), dtype=target.dtype if hasattr(target, "dtype") else np_backend.dtype)
    if arr.size != target.size:
        raise ValueError(f"Array size mismatch: expected {target.size}, got {arr.size}")
    target[...] = arr.reshape(target.shape) if hasattr(target, "shape") else arr


def _serialize_param_groups(param_groups) -> list[dict[str, float]]:
    out = []
    for group in param_groups:
        entry = {"lr": float(group["lr"])}
        if "weight_decay" in group:
            entry["weight_decay"] = float(group["weight_decay"])
        out.append(entry)
    return out


def _apply_param_groups(param_groups, saved_groups: list[dict[str, float]]) -> None:
    if len(param_groups) != len(saved_groups):
        raise ValueError(
            f"Optimizer param group count mismatch: expected {len(param_groups)}, got {len(saved_groups)}"
        )
    for group, saved in zip(param_groups, saved_groups):
        group["lr"] = float(saved["lr"])
        if "weight_decay" in saved:
            group["weight_decay"] = float(saved["weight_decay"])


def _optimizer_state_dict(optimizer) -> tuple[dict[str, Any], dict[str, host_np.ndarray]]:
    from NimbleML.optimizers.adam import Adam
    from NimbleML.optimizers.nag import NAG
    from NimbleML.optimizers.rmsprop import RMSProp
    from NimbleML.optimizers.sgdm import SGDM

    cls = type(optimizer).__name__
    meta: dict[str, Any] = {
        "class": cls,
        "param_groups": _serialize_param_groups(optimizer.param_groups),
        "num_params": len(optimizer.params),
    }
    arrays: dict[str, host_np.ndarray] = {}

    if isinstance(optimizer, Adam):
        meta.update(
            {
                "t": int(optimizer.t),
                "beta1": float(optimizer.beta1),
                "beta2": float(optimizer.beta2),
                "epsilon": float(optimizer.epsilon),
                "weight_decay": float(optimizer.weight_decay),
            }
        )
        for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
            arrays[f"optimizer/m/{i}"] = _to_host_ndarray(m)
            arrays[f"optimizer/v/{i}"] = _to_host_ndarray(v)
        for i, master in enumerate(getattr(optimizer, "masters", []) or []):
            if master is not None:
                arrays[f"optimizer/master/{i}"] = _to_host_ndarray(master)
    elif isinstance(optimizer, (SGDM, NAG)):
        meta["momentum"] = float(optimizer.momentum)
        for i, velocity in enumerate(optimizer.velocities):
            arrays[f"optimizer/velocity/{i}"] = _to_host_ndarray(velocity)
    elif isinstance(optimizer, RMSProp):
        meta["rho"] = float(optimizer.rho)
        meta["epsilon"] = float(optimizer.epsilon)
        for i, sq_avg in enumerate(optimizer.sq_grad_avg):
            arrays[f"optimizer/sq_grad_avg/{i}"] = _to_host_ndarray(sq_avg)
    elif cls == "SGD":
        pass
    else:
        raise TypeError(f"Unsupported optimizer type for checkpoint: {cls}")

    return meta, arrays


def _load_optimizer_state(optimizer, meta: dict[str, Any], data: host_np.lib.npyio.NpzFile) -> None:
    from NimbleML.optimizers.adam import Adam
    from NimbleML.optimizers.nag import NAG
    from NimbleML.optimizers.rmsprop import RMSProp
    from NimbleML.optimizers.sgdm import SGDM

    cls = type(optimizer).__name__
    if meta["class"] != cls:
        raise ValueError(f"Optimizer class mismatch: checkpoint has {meta['class']!r}, model has {cls!r}")
    if meta["num_params"] != len(optimizer.params):
        raise ValueError(
            f"Optimizer parameter count mismatch: checkpoint has {meta['num_params']}, model has {len(optimizer.params)}"
        )

    _apply_param_groups(optimizer.param_groups, meta["param_groups"])

    if isinstance(optimizer, Adam):
        optimizer.t = int(meta["t"])
        for key in ("beta1", "beta2", "epsilon", "weight_decay"):
            if float(getattr(optimizer, key)) != float(meta[key]):
                raise ValueError(f"Adam hyperparameter mismatch for {key}")
        for i in range(len(optimizer.params)):
            _load_ndarray_into(optimizer.m[i], data[f"optimizer/m/{i}"])
            _load_ndarray_into(optimizer.v[i], data[f"optimizer/v/{i}"])
        masters = getattr(optimizer, "masters", None)
        if masters is not None:
            for i, master in enumerate(masters):
                if master is None:
                    continue
                key = f"optimizer/master/{i}"
                if key in data.files:
                    _load_ndarray_into(master, data[key])
                else:
                    # Older checkpoint without masters: re-seed from the
                    # (already loaded) model weights.
                    param = optimizer.params[i]
                    master[...] = np.asarray(param.data, dtype=master.dtype).reshape(-1)
    elif isinstance(optimizer, (SGDM, NAG)):
        if float(optimizer.momentum) != float(meta["momentum"]):
            raise ValueError("SGDM/NAG momentum mismatch")
        for i in range(len(optimizer.params)):
            _load_ndarray_into(optimizer.velocities[i], data[f"optimizer/velocity/{i}"])
    elif isinstance(optimizer, RMSProp):
        if float(optimizer.rho) != float(meta["rho"]) or float(optimizer.epsilon) != float(meta["epsilon"]):
            raise ValueError("RMSProp hyperparameter mismatch")
        for i in range(len(optimizer.params)):
            _load_ndarray_into(optimizer.sq_grad_avg[i], data[f"optimizer/sq_grad_avg/{i}"])
    elif cls != "SGD":
        raise TypeError(f"Unsupported optimizer type for checkpoint: {cls}")


def _scheduler_state_dict(scheduler) -> dict[str, Any]:
    from NimbleML.optimizers.schedulers.cosine_annealing import CosineAnnealing
    from NimbleML.optimizers.schedulers.linear_warmup import LinearWarmup
    from NimbleML.optimizers.schedulers.step_lr import StepLR

    cls = type(scheduler).__name__
    meta: dict[str, Any] = {
        "class": cls,
        "last_epoch": int(scheduler.last_epoch),
        "base_lrs": [float(lr) for lr in scheduler.base_lrs],
    }

    if isinstance(scheduler, StepLR):
        meta["step_size"] = int(scheduler.step_size)
        meta["gamma"] = float(scheduler.gamma)
    elif isinstance(scheduler, CosineAnnealing):
        meta["T_max"] = int(scheduler.T_max)
        meta["eta_min"] = float(scheduler.eta_min)
    elif isinstance(scheduler, LinearWarmup):
        meta["warmup_steps"] = int(scheduler.warmup_steps)
        meta["start_factor"] = float(scheduler.start_factor)
        meta["inner"] = _scheduler_state_dict(scheduler.inner_scheduler)
    else:
        raise TypeError(f"Unsupported scheduler type for checkpoint: {cls}")

    return meta


def _load_scheduler_state(scheduler, meta: dict[str, Any]) -> None:
    from NimbleML.optimizers.schedulers.cosine_annealing import CosineAnnealing
    from NimbleML.optimizers.schedulers.linear_warmup import LinearWarmup
    from NimbleML.optimizers.schedulers.step_lr import StepLR

    cls = type(scheduler).__name__
    if meta["class"] != cls:
        raise ValueError(f"Scheduler class mismatch: checkpoint has {meta['class']!r}, model has {cls!r}")

    scheduler.last_epoch = int(meta["last_epoch"])
    scheduler.base_lrs = [float(lr) for lr in meta["base_lrs"]]

    if isinstance(scheduler, StepLR):
        if int(scheduler.step_size) != int(meta["step_size"]) or float(scheduler.gamma) != float(meta["gamma"]):
            raise ValueError("StepLR config mismatch")
    elif isinstance(scheduler, CosineAnnealing):
        if int(scheduler.T_max) != int(meta["T_max"]) or float(scheduler.eta_min) != float(meta["eta_min"]):
            raise ValueError("CosineAnnealing config mismatch")
    elif isinstance(scheduler, LinearWarmup):
        if int(scheduler.warmup_steps) != int(meta["warmup_steps"]) or float(scheduler.start_factor) != float(
            meta["start_factor"]
        ):
            raise ValueError("LinearWarmup config mismatch")
        _load_scheduler_state(scheduler.inner_scheduler, meta["inner"])
    else:
        raise TypeError(f"Unsupported scheduler type for checkpoint: {cls}")

    scheduler.optimizer.set_lr(scheduler.get_lr())


def _model_arrays(model) -> dict[str, host_np.ndarray]:
    return {f"{_MODEL_PREFIX}{name}": _to_host_array(param) for name, param in named_parameters(model)}


def _load_model_arrays(model, data: host_np.lib.npyio.NpzFile) -> None:
    expected = dict(named_parameters(model))
    if not expected:
        raise ValueError("Model has no parameters to load.")

    prefixed = {name: f"{_MODEL_PREFIX}{name}" for name in expected}
    use_prefix = any(key in data.files for key in prefixed.values())

    for name, param in expected.items():
        key = prefixed[name] if use_prefix else name
        if key not in data.files:
            raise ValueError(f"Checkpoint missing parameter {name!r}")
        loaded = host_np.asarray(data[key], dtype=param.data.dtype).reshape(param.shape)
        if loaded.size != param.size:
            raise ValueError(f"Shape mismatch for {name}: expected {param.shape}, got {loaded.shape}")
        param.data[...] = np.asarray(loaded.ravel(), dtype=param.data.dtype)


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
        if _META_KEY in data.files:
            raise ValueError("File is a training checkpoint; use load_checkpoint() instead of load().")
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


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model weights and optional optimizer / scheduler state for resume."""
    arrays = _model_arrays(model)
    if not arrays:
        raise ValueError("Model has no parameters to save.")

    meta: dict[str, Any] = {"version": _CHECKPOINT_VERSION, "step": step, "extra": extra or {}}
    if optimizer is not None:
        opt_meta, opt_arrays = _optimizer_state_dict(optimizer)
        meta["optimizer"] = opt_meta
        arrays.update(opt_arrays)
    if scheduler is not None:
        meta["scheduler"] = _scheduler_state_dict(scheduler)

    arrays[_META_KEY] = host_np.array(json.dumps(meta))
    host_np.savez(path, **arrays)


def load_checkpoint(path, model, optimizer=None, scheduler=None) -> dict[str, Any]:
    """Restore a checkpoint saved by :func:`save_checkpoint`.

    Returns metadata dict with at least ``step`` and ``extra`` keys.
    """
    with host_np.load(path) as data:
        if _META_KEY not in data.files:
            raise ValueError("Not a training checkpoint (missing metadata); use load() for weights-only files.")

        meta = json.loads(str(data[_META_KEY]))
        if meta.get("version") != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {meta.get('version')}")

        _load_model_arrays(model, data)

        if optimizer is not None:
            if "optimizer" not in meta:
                raise ValueError("Checkpoint does not contain optimizer state.")
            _load_optimizer_state(optimizer, meta["optimizer"], data)
        elif "optimizer" in meta:
            raise ValueError("Checkpoint contains optimizer state but no optimizer was provided.")

        if scheduler is not None:
            if "scheduler" not in meta:
                raise ValueError("Checkpoint does not contain scheduler state.")
            _load_scheduler_state(scheduler, meta["scheduler"])
        elif "scheduler" in meta:
            raise ValueError("Checkpoint contains scheduler state but no scheduler was provided.")

    return {"step": meta.get("step"), "extra": meta.get("extra") or {}}
