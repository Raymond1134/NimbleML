"""Autograd tensor with NumPy/CuPy backend"""
from __future__ import annotations
import os
from math import prod
from . import np_backend
from .axis import normalize_axis, normalize_axes
from .np_backend import np

_VALIDATE_TENSORS = os.environ.get("NIMBLEML_TENSOR_VALIDATE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _NOOP_BACKWARD():
    return None


_EMPTY_PREV = frozenset()

_SAVE_FOR_BACKWARD_STATS = {"copies": 0, "aliases": 0}


def save_for_backward_stats() -> dict:
    """Return counts of copy vs alias saves since last reset."""
    return dict(_SAVE_FOR_BACKWARD_STATS)


def reset_save_for_backward_stats() -> None:
    """Reset :func:`save_for_backward_stats` counters."""
    _SAVE_FOR_BACKWARD_STATS["copies"] = 0
    _SAVE_FOR_BACKWARD_STATS["aliases"] = 0


def _is_stable_parameter_view(tensor) -> bool:
    """Leaf parameters (and weight transposes) are safe to alias until ``optimizer.step()``."""
    if not isinstance(tensor, Tensor):
        return False
    if tensor._op == "":
        return True
    if tensor._op == "transpose" and len(tensor._prev) == 1:
        parent = next(iter(tensor._prev))
        return parent._op == ""
    return False


def _is_stable_matmul_operand(tensor):
    return _is_stable_parameter_view(tensor)


def _save_for_backward(arr, tensor=None):
    """Own-memory copy for autograd closures (avoids stale CuPy pool views).

    When *tensor* is a stable parameter view, returns a contiguous alias instead
    of copying — parameter storage is not overwritten until the optimizer step.
    """
    if tensor is not None and _is_stable_parameter_view(tensor):
        _SAVE_FOR_BACKWARD_STATS["aliases"] += 1
        view = np.asarray(arr)
        if view.flags.c_contiguous:
            return view
        return np.ascontiguousarray(view)
    _SAVE_FOR_BACKWARD_STATS["copies"] += 1
    return np.asarray(arr).copy()


def _grad_out(tensor, shape=None):
    """Upstream gradient at backward time (copy-free when exclusively owned).

    When ``tensor._grad_fresh`` is set by :meth:`Tensor._accumulate_grad`, the
    grad buffer is exclusively owned and safe to alias (no CuPy pool reuse risk).
    Falls back to a copy when the grad was already handed out or is not contiguous.
    """
    if tensor.grad is None:
        return None
    g = Tensor._asarray(tensor.grad)
    if shape is not None:
        g = g.reshape(shape)
    if getattr(tensor, "_grad_fresh", False) and g.flags.c_contiguous:
        tensor._grad_fresh = False
        _SAVE_FOR_BACKWARD_STATS["aliases"] += 1
        return g
    _SAVE_FOR_BACKWARD_STATS["copies"] += 1
    return np.asarray(g, dtype=np_backend.dtype).copy()


def _build_backward_topo(root: Tensor) -> list[Tensor]:
    """Post-order DFS from *root*."""
    count_visited: set[Tensor] = set()
    count_stack: list[Tensor] = [root]
    while count_stack:
        node = count_stack.pop()
        if node in count_visited:
            continue
        count_visited.add(node)
        count_stack.extend(node._prev)

    visited: set[Tensor] = set()
    topo: list[Tensor] = [None] * len(count_visited)  # type: ignore[list-item]
    write = 0
    stack: list[tuple[Tensor, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if not expanded:
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            prev_list = list(node._prev)
            for child in reversed(prev_list):
                if child not in visited:
                    stack.append((child, False))
        else:
            topo[write] = node
            write += 1

    return topo


def _matmul_right_view(other, right_shape, save_right):
    if save_right is not None:
        return save_right.reshape(right_shape)
    if other._op == "transpose" and len(other._prev) == 1:
        parent = next(iter(other._prev))
        if parent._op == "":
            parent_arr = parent._view()
            return np.swapaxes(parent_arr, -2, -1)
    return other._view(right_shape)


class Tensor:
    """Autograd tensor wrapping a flat NumPy/CuPy buffer."""

    @staticmethod
    def _as_int64(data):
        from .np_backend import as_int64

        return as_int64(data)

    @staticmethod
    def _size_for_shape(shape: tuple) -> int:
        return prod(shape) if shape else 1

    @staticmethod
    def _coerce_flat_buffer(data):
        """Return a 1-D compute-dtype buffer, avoiding redundant copies when possible."""
        if isinstance(data, (int, float)):
            return np.array([float(data)], dtype=np_backend.dtype)
        if isinstance(data, np.ndarray) and data.ndim == 1 and data.dtype == np_backend.dtype:
            return data
        return np.asarray(data, dtype=np_backend.dtype).ravel()

    def _init_common(
        self,
        *,
        data,
        shape: tuple,
        requires_grad: bool,
        _children,
        _op: str,
        is_view: bool,
    ) -> None:
        self.data = data
        self.shape = shape
        self._size = self._size_for_shape(shape)
        self._is_view = is_view
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fresh = False
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._backward_topo_cache = None
        if _VALIDATE_TENSORS:
            self._validate()

    @classmethod
    def _make_view(cls, parent: Tensor, shape: tuple, *, _op: str) -> Tensor:
        """Tensor sharing *parent*'s flat storage (reshape / squeeze / unsqueeze)."""
        out = cls.__new__(cls)
        out._init_common(
            data=parent.data,
            shape=shape,
            requires_grad=parent.requires_grad,
            _children=(parent,),
            _op=_op,
            is_view=True,
        )
        return out

    @classmethod
    def from_int64(cls, data, shape):
        """Integer token/index tensor (int64 on the active backend, no grad)."""
        arr = cls._as_int64(data).reshape(-1)
        expected = cls._size_for_shape(shape)
        if _VALIDATE_TENSORS and arr.size != expected:
            raise ValueError("Shape does not match number of int64 elements.")
        obj = cls.__new__(cls)
        obj.data = arr
        obj.shape = shape
        obj._size = expected
        obj._is_view = False
        obj.requires_grad = False
        obj.grad = None
        obj._grad_fresh = False
        obj._backward = lambda: None
        obj._prev = set()
        obj._op = ""
        obj._backward_topo_cache = None
        if _VALIDATE_TENSORS:
            obj._validate()
        return obj

    @staticmethod
    def _is_int64_tensor(tensor):
        return getattr(tensor.data, "dtype", None) == np.int64

    @staticmethod
    def _asarray(data):
        if getattr(data, "dtype", None) == np.int64:
            return np.asarray(data)
        return np.asarray(data, dtype=np_backend.dtype)

    def _view(self, shape=None):
        """Shaped view of flat ``self.data`` without redundant ``asarray`` dispatch."""
        if shape is None:
            shape = self.shape
        return self.data.reshape(shape) if shape else self.data.reshape(())

    def __init__(self, data, shape, requires_grad=False, _children=(), _op=""):
        flat = self._coerce_flat_buffer(data)
        self._init_common(
            data=flat,
            shape=shape,
            requires_grad=requires_grad,
            _children=_children,
            _op=_op,
            is_view=False,
        )

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __getitem__(self, indices):
        arr = self._view() if self.shape else self.data.reshape(())
        return float(arr[indices])

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Number of elements (cached at construction)."""
        return self._size

    @property
    def T(self):
        """Alias for transpose() on 2D tensors."""
        return self.transpose()

    def item(self):
        """Return the single element as a Python float."""
        if self.size != 1:
            raise ValueError("Only scalar tensors can be converted to a Python scalar.")
        return float(self.data[0])

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients. With ``set_to_none=True``, drop refs instead of zero-fill."""
        if set_to_none:
            self.grad = None
            self._grad_fresh = False
        else:
            gdt = getattr(self, "_grad_dtype", None) or np_backend.dtype
            self.grad = np.zeros(self.size, dtype=gdt)
            self._grad_fresh = True

    def detach(self):
        """Return a tensor sharing storage but disconnected from the graph."""
        out = Tensor(self.data, self.shape, requires_grad=False)
        return out

    def backward(self, grad=None, retain_graph=False):
        """Run reverse-mode autodiff from this tensor.

        Unless ``retain_graph=True``, each node's ``_backward`` closure is
        released once it has run. Those closures capture their own output
        tensor (and the activations needed for the backward), forming reference
        cycles that Python's refcounting cannot reclaim on its own. Clearing
        them lets the graph free immediately instead of lingering until the
        cyclic GC runs -- without this, a fast training loop piles up graphs and
        exhausts GPU memory.
        """
        if grad is None:
            if self.size != 1:
                raise ValueError("grad must be specified for non-scalar tensors.")
            grad = np.array([1.0], dtype=np_backend.dtype)
        elif isinstance(grad, (int, float)):
            grad = np.array([float(grad)], dtype=np_backend.dtype)
        elif isinstance(grad, Tensor):
            grad = grad.data
        else:
            grad = np.asarray(grad, dtype=np_backend.dtype)

        self._accumulate_grad(grad)

        topo = self._backward_topo_cache
        if topo is None:
            topo = _build_backward_topo(self)
            if retain_graph:
                self._backward_topo_cache = topo

        # Native autograd runner: topo + reverse execute of Python backward closures.
        from NimbleML._native_loader import native

        native.autograd_clear()
        id_map: dict = {}
        for node in topo:
            parent_ids = [id_map[p] for p in node._prev if p in id_map]
            bn = node._backward
            nid = native.autograd_add_py_node(list(parent_ids), bool(node.requires_grad), bn)
            id_map[node] = nid
        native.autograd_run_backward(id_map[self], bool(retain_graph))

        if not retain_graph:
            for node in topo:
                node._backward = _NOOP_BACKWARD
                node._prev = _EMPTY_PREV
            self._backward_topo_cache = None
    
    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return
        # Parameters may pin a wider grad dtype (fp32 under fp16 compute) so
        # microbatch accumulation does not underflow.
        gdt = getattr(self, "_grad_dtype", None) or np_backend.dtype
        grad = np.asarray(grad).reshape(-1)
        if grad.dtype != gdt:
            grad = grad.astype(gdt)
            copied = True
        else:
            copied = False
        if self.grad is None:
            self.grad = grad if copied else grad.copy()
        else:
            self.grad = self.grad + grad
        self._grad_fresh = True

    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(np.array([other], dtype=np_backend.dtype), (), requires_grad=False)

    def __add__(self, other):
        return self._apply_binary(
            other,
            np.add,
            lambda grad, a, b: grad,
            lambda grad, a, b: grad,
            "add",
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_binary(
            other,
            np.subtract,
            lambda grad, a, b: grad,
            lambda grad, a, b: -grad,
            "sub",
        )

    def __rsub__(self, other):
        return self._ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        return self._apply_binary(
            other,
            np.multiply,
            lambda grad, a, b: grad * b,
            lambda grad, a, b: grad * a,
            "mul",
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_binary(
            other,
            np.divide,
            lambda grad, a, b: grad / b,
            lambda grad, a, b: -(grad * a) / (b * b),
            "div",
        )

    def __rtruediv__(self, other):
        return self._ensure_tensor(other).__truediv__(self)

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            raise NotImplementedError("Tensor ** Tensor is not supported yet.")
        exponent = float(exponent)

        shape = self.shape
        a = _save_for_backward(self._view(shape) if shape else self.data)
        out_data = np.power(a, exponent)

        out = Tensor(
            out_data.ravel(),
            shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="pow",
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_out = _grad_out(out, shape)
            grad_a = grad_out * exponent * np.power(a, exponent - 1.0)
            self._accumulate_grad(grad_a.ravel())

        out._backward = _backward
        return out

    def sqrt(self):
        """Element-wise square root."""
        return self ** 0.5

    def __matmul__(self, other):
        return self.matmul(other)

    def _apply_binary(self, other, op, grad_a_rule, grad_b_rule, op_name):
        other = self._ensure_tensor(other)
        a = self
        b = other
        out_shape, shape_a, shape_b = self._broadcast_shape(a.shape, b.shape)
        a_np = a._view(shape_a)
        b_np = b._view(shape_b)
        # Only copy operands the backward rule reads (mul/div/pow). Add/sub only
        # pass grad through and do not need saved inputs; copying every residual
        # add would ~double activation memory on deep models.
        needs_a = a.requires_grad and op_name in ("mul", "div", "pow")
        needs_b = b.requires_grad and op_name in ("mul", "div", "pow")
        save_a = _save_for_backward(a_np) if needs_a else a_np
        save_b = _save_for_backward(b_np) if needs_b else b_np
        out_data = op(a_np, b_np).ravel()

        out = Tensor(out_data, out_shape, requires_grad=a.requires_grad or b.requires_grad, _children=(a, b), _op=op_name)

        def _backward():
            if out.grad is None:
                return
            grad_out = _grad_out(out, out_shape)
            grad_a = grad_a_rule(grad_out, save_a, save_b)
            grad_b = grad_b_rule(grad_out, save_a, save_b)
            if a.requires_grad:
                a._accumulate_grad(Tensor._reduce_broadcast_grad(grad_a, a.shape))
            if b.requires_grad:
                b._accumulate_grad(Tensor._reduce_broadcast_grad(grad_b, b.shape))

        out._backward = _backward
        return out

    @staticmethod
    def _reduce_broadcast_grad(grad, shape):
        grad = np.asarray(grad, dtype=np_backend.dtype)
        if grad.ndim == 0:
            return np.array([grad.item()], dtype=np_backend.dtype)

        ndim = grad.ndim
        target_ndim = len(shape)
        padded = Tensor._pad_shape(shape, ndim)
        axes = tuple(i for i, (dim, target) in enumerate(zip(grad.shape, padded)) if target == 1 and dim != 1)

        if axes:
            grad = np.sum(grad, axis=axes, keepdims=True)

        leading = ndim - target_ndim
        if leading > 0:
            grad = np.sum(grad, axis=tuple(range(leading)))

        return grad.reshape(-1)
    
    @staticmethod
    def _pad_shape(shape, target_ndim):
        if len(shape) > target_ndim:
            raise ValueError("Target ndim must be greater than or equal to the tensor's current ndim.")
        return (1,) * (target_ndim - len(shape)) + shape
    
    @staticmethod
    def _broadcast_shape(shape_a, shape_b):
        ndim = max(len(shape_a), len(shape_b))
        shape_a = Tensor._pad_shape(shape_a, ndim)
        shape_b = Tensor._pad_shape(shape_b, ndim)
        out = []
        
        for dim_a, dim_b in zip(shape_a, shape_b):
            if dim_a == dim_b or dim_a == 1 or dim_b == 1:
                out.append(max(dim_a, dim_b))
            else:
                raise ValueError(f"Shapes {shape_a} and {shape_b} are not compatible for broadcasting.")

        return tuple(out), shape_a, shape_b

    def matmul(self, other):
        """
        Matrix multiply via np.matmul (supports batched leading dims).

        On GPU, ``np`` is CuPy and ``matmul`` dispatches to cuBLAS GEMM.

        Examples: (B, C) @ (C, D), (B, T, C) @ (C, D), (m, k) @ (k, n).
        """
        other = self._ensure_tensor(other)
        left_shape = self.shape
        right_shape = other.shape

        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("Matrix multiplication requires tensors with at least one dimension.")

        left = np.ascontiguousarray(self._view(left_shape))
        right = np.ascontiguousarray(other._view(right_shape))

        save_left = _save_for_backward(left, tensor=self) if other.requires_grad else None
        save_right = (
            _save_for_backward(right, tensor=other)
            if self.requires_grad and not _is_stable_matmul_operand(other)
            else None
        )

        try:
            out_arr = np.matmul(left, right)
        except ValueError as exc:
            raise ValueError(f"matmul shapes {left_shape} and {right_shape} are not compatible.") from exc

        out_shape = out_arr.shape
        out = Tensor(
            out_arr.ravel(),
            out_shape,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward():
            if out.grad is None:
                return

            grad_out = _grad_out(out, out_shape)
            left_arr = save_left.reshape(left_shape) if save_left is not None else None

            if self.requires_grad:
                right_arr = _matmul_right_view(other, right_shape, save_right)
                if right_arr.ndim == 1:
                    grad_left = np.matmul(grad_out, right_arr)
                else:
                    right_T = np.ascontiguousarray(np.swapaxes(right_arr, -2, -1))
                    grad_left = np.matmul(grad_out, right_T)
                self._accumulate_grad(grad_left.ravel())

            if other.requires_grad:
                if left_arr.ndim == 1:
                    grad_right = left_arr[:, None] * grad_out[..., None, :]
                elif len(right_shape) == 1:
                    grad_right = np.tensordot(
                        left_arr, grad_out,
                        axes=(list(range(left_arr.ndim - 1)), list(range(grad_out.ndim))),
                    )

                elif len(right_shape) == 2:
                    contract_axes = (list(range(left_arr.ndim - 1)), list(range(grad_out.ndim - 1)))
                    grad_right = np.tensordot(left_arr, grad_out, axes=contract_axes)
                else:
                    left_T = np.ascontiguousarray(np.swapaxes(left_arr, -2, -1))
                    grad_right = np.matmul(left_T, grad_out)
                other._accumulate_grad(grad_right.ravel())

        out._backward = _backward
        return out

    def relu(self):
        """Rectified linear unit."""
        arr = _save_for_backward(self.data)
        out_data = np.maximum(arr, 0.0)
        out = Tensor(out_data, self.shape, requires_grad=self.requires_grad, _children=(self,), _op="relu")
        mask = (arr > 0).astype(np_backend.dtype)

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_out = _grad_out(out, self.shape)
            self._accumulate_grad(grad_out * mask)

        out._backward = _backward
        return out

    def gelu(self):
        """GELU activation (tanh approximation, GPU-safe)."""
        from NimbleML.utils.activations import gelu_backward, gelu_forward

        arr = self._view() if self.shape else self.data
        save_pre = _save_for_backward(arr)
        out_data, tanh_u = gelu_forward(save_pre)
        save_tanh_u = _save_for_backward(tanh_u)
        out = Tensor(
            out_data.ravel(),
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="gelu",
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_out = _grad_out(out, save_pre.shape)
            grad = gelu_backward(grad_out, save_pre, save_tanh_u)
            self._accumulate_grad(grad.ravel())

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Sum reduction along axis, or over all elements."""
        if axis is None:
            reduce_axes = None
        else:
            if isinstance(axis, int):
                axis = (axis,)
            elif isinstance(axis, (list, tuple)):
                axis = tuple(axis)
            else:
                raise TypeError("axis must be int, tuple, list, or None")

            reduce_axes = normalize_axes(self.ndim, axis)

        arr = self.data.reshape(self.shape) if self.shape else self.data.reshape(())
        out_arr = np.sum(arr, axis=reduce_axes, keepdims=keepdims)
        out_shape = out_arr.shape
        out = Tensor(out_arr.ravel(), out_shape, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            if self.ndim == 0:
                self._accumulate_grad(_grad_out(out, (1,)))
                return
            grad_out = _grad_out(out, out_shape)
            if reduce_axes is not None and not keepdims:
                expand_shape = list(self.shape)
                for ax in reduce_axes:
                    expand_shape[ax] = 1
                grad_out = grad_out.reshape(expand_shape)
            grad_in = np.broadcast_to(grad_out, self.shape).ravel()
            self._accumulate_grad(grad_in)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean reduction along axis, or over all elements."""
        summed_tensor = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = float(self.size)
        else:
            axes = normalize_axes(self.ndim, axis)
            count = float(prod(self.shape[ax] for ax in axes))
        return summed_tensor / count

    def reshape(self, new_shape):
        """Return a view with a new shape (same storage)."""
        new_shape = tuple(new_shape) if not isinstance(new_shape, tuple) else new_shape
        if self._size_for_shape(new_shape) != self._size:
            raise ValueError("New shape must have the same number of elements as the original shape.")

        out = Tensor._make_view(self, new_shape, _op="reshape")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_out = _grad_out(out, new_shape)
            self._accumulate_grad(grad_out)

        out._backward = _backward
        return out

    def transpose(self):
        """Transpose a 2D tensor."""
        if self.ndim != 2:
            raise ValueError("Transpose requires a 2D tensor.")

        rows, cols = self.shape
        out_data = self._view().T.ravel()
        out = Tensor(
            out_data,
            (cols, rows),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_out = _grad_out(out, (cols, rows))
            self._accumulate_grad(np.ascontiguousarray(np.swapaxes(grad_out, -2, -1)).ravel())

        out._backward = _backward
        return out

    def _validate(self):
        if not isinstance(self.shape, tuple):
            raise ValueError("Shape must be a tuple.")
        if len(self.data) != self._size:
            raise ValueError(
                "Data size does not match the specified shape. "
                f"Expected {self._size} data values, got {len(self.data)} data values."
            )

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten dimensions from start_dim through end_dim."""
        if self.ndim == 0:
            if start_dim not in (0, -1) or end_dim not in (0, -1):
                raise ValueError("start_dim and end_dim must be 0 or -1 for scalar tensors.")
            return self.reshape((1,))

        start_dim = normalize_axis(self.ndim, start_dim)
        end_dim = normalize_axis(self.ndim, end_dim)

        if start_dim > end_dim:
            raise ValueError("start_dim must be less than or equal to end_dim.")

        flattened = prod(self.shape[start_dim:end_dim + 1])
        new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
        return self.reshape(new_shape)
    
    def squeeze(self, axis=None):
        """Remove dimensions of size 1."""
        if axis is None:
            new_shape = tuple(dim for dim in self.shape if dim != 1)
            return self.reshape(new_shape if new_shape else ())
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        else:
            raise TypeError("axis must be int, tuple, list, or None")
        
        axes = normalize_axes(self.ndim, axis)
        if len(set(axes)) != len(axes):
            raise ValueError("axis has duplicates")
        
        for ax in axes:
            if self.shape[ax] != 1:
                raise ValueError(f"Cannot squeeze dimension {ax} with size {self.shape[ax]}")
        
        new_shape = tuple(dim for i, dim in enumerate(self.shape) if i not in axes)
        return self.reshape(new_shape if new_shape else ())

    def unsqueeze(self, axis):
        """Insert a size-1 dimension at axis."""
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        else:
            raise TypeError("axis must be int, tuple, or list")

        new_ndim = self.ndim + len(axis)
        axes = normalize_axes(new_ndim, axis)
        if len(set(axes)) != len(axes):
            raise ValueError("axis has duplicates")

        axes_set = set(axes)
        new_shape = []
        src_i = 0
        for i in range(new_ndim):
            if i in axes_set:
                new_shape.append(1)
            else:
                new_shape.append(self.shape[src_i])
                src_i += 1

        return self.reshape(tuple(new_shape))
