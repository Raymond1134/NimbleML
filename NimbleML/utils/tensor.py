"""Autograd tensor with NumPy/CuPy backend"""
from math import prod
from . import np_backend
from .np_backend import np

class Tensor:
    """Public class Tensor."""

    @staticmethod
    def _asarray(data):
        return np.asarray(data, dtype=np_backend.dtype)

    def __init__(self, data, shape, requires_grad=False, _children=(), _op=""):
        if isinstance(data, (int, float)):
            data = np.array([float(data)], dtype=np_backend.dtype)
        self.data = np.asarray(data, dtype=np_backend.dtype).ravel()
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._validate()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __getitem__(self, indices):
        arr = Tensor._asarray(self.data).reshape(self.shape) if self.shape else Tensor._asarray(self.data).reshape(())
        return float(arr[indices])

    @property
    def ndim(self):
        """Public function ndim."""
        return len(self.shape)

    @property
    def size(self):
        """Public function size."""
        return prod(self.shape) if self.shape else 1

    @property
    def T(self):
        """Public function T."""
        return self.transpose()

    def item(self):
        """Public function item."""
        if self.size != 1:
            raise ValueError("Only scalar tensors can be converted to a Python scalar.")
        return float(self.data[0])

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients. With ``set_to_none=True``, drop refs instead of zero-fill."""
        if set_to_none:
            self.grad = None
        else:
            self.grad = np.zeros(self.size, dtype=np_backend.dtype)

    def backward(self, grad=None):
        """Public function backward."""
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

        topo = []
        visited = set()

        def build(node):
            """Public function build."""
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        for node in reversed(topo):
            node._backward()
    
    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return
        grad = np.asarray(grad, dtype=np_backend.dtype)
        if self.grad is None:
            self.grad = np.array(grad, dtype=np_backend.dtype)
        else:
            self.grad += grad

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
        a = Tensor._asarray(self.data).reshape(shape)
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
            grad_out = out.grad.reshape(shape)
            grad_a = grad_out * exponent * np.power(a, exponent - 1.0)
            self._accumulate_grad(grad_a.ravel())

        out._backward = _backward
        return out

    def sqrt(self):
        """Public function sqrt."""
        return self ** 0.5

    def __matmul__(self, other):
        return self.matmul(other)

    def _apply_binary(self, other, op, grad_a_rule, grad_b_rule, op_name):
        other = self._ensure_tensor(other)
        a = self
        b = other
        out_shape, shape_a, shape_b = self._broadcast_shape(a.shape, b.shape)
        a_np = Tensor._asarray(a.data).reshape(shape_a)
        b_np = Tensor._asarray(b.data).reshape(shape_b)
        out_data = op(a_np, b_np).ravel()

        out = Tensor(out_data, out_shape, requires_grad=a.requires_grad or b.requires_grad, _children=(a, b), _op=op_name)

        def _backward():
            if out.grad is None:
                return
            grad_out = out.grad.reshape(out_shape)
            grad_a = grad_a_rule(grad_out, a_np, b_np)
            grad_b = grad_b_rule(grad_out, a_np, b_np)
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

        left = Tensor._asarray(self.data).reshape(left_shape)
        right = Tensor._asarray(other.data).reshape(right_shape)

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

            grad_out = Tensor._asarray(out.grad).reshape(out_shape)
            left_arr = Tensor._asarray(self.data).reshape(left_shape)
            right_arr = Tensor._asarray(other.data).reshape(right_shape)

            if self.requires_grad:
                if right_arr.ndim == 1:
                    grad_left = np.matmul(grad_out, right_arr)
                else:
                    right_T = np.swapaxes(right_arr, -2, -1)
                    grad_left = np.matmul(grad_out, right_T)
                self._accumulate_grad(grad_left.ravel())

            if other.requires_grad:
                if left_arr.ndim == 1:
                    grad_right = np.matmul(left_arr, grad_out)
                elif right_arr.ndim == 1:
                    grad_right = np.matmul(left_arr[..., np.newaxis], grad_out)
                elif right_arr.ndim == 2:
                    contract_axes = (list(range(left_arr.ndim - 1)), list(range(grad_out.ndim - 1)))
                    grad_right = np.tensordot(left_arr, grad_out, axes=contract_axes)
                else:
                    left_T = np.swapaxes(left_arr, -2, -1)
                    grad_right = np.matmul(left_T, grad_out)
                other._accumulate_grad(grad_right.ravel())

        out._backward = _backward
        return out

    def relu(self):
        """Public function relu."""
        arr = self.data
        out_data = np.maximum(arr, 0.0)
        out = Tensor(out_data, self.shape, requires_grad=self.requires_grad, _children=(self,), _op="relu")
        mask = (arr > 0).astype(np_backend.dtype)

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * mask)

        out._backward = _backward
        return out

    def gelu(self):
        """GELU activation (tanh approximation, GPU-safe)."""
        from NimbleML.activations.gelu import gelu_backward, gelu_forward

        arr = Tensor._asarray(self.data).reshape(self.shape) if self.shape else Tensor._asarray(self.data)
        out_data, tanh_u = gelu_forward(arr)
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
            grad_out = Tensor._asarray(out.grad).reshape(arr.shape)
            grad = gelu_backward(grad_out, arr, tanh_u)
            self._accumulate_grad(grad.ravel())

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Public function sum."""
        if axis is None:
            reduce_axes = None
        else:
            if isinstance(axis, int):
                axis = (axis,)
            elif isinstance(axis, (list, tuple)):
                axis = tuple(axis)
            else:
                raise TypeError("axis must be int, tuple, list, or None")

            axis = tuple(ax + self.ndim if ax < 0 else ax for ax in axis)
            for ax in axis:
                if ax < 0 or ax >= self.ndim:
                    raise ValueError(f"axis {ax} out of range for ndim {self.ndim}")
            reduce_axes = axis

        arr = self.data.reshape(self.shape) if self.shape else self.data.reshape(())
        out_arr = np.sum(arr, axis=reduce_axes, keepdims=keepdims)
        out_shape = out_arr.shape
        out = Tensor(out_arr.ravel(), out_shape, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            if self.ndim == 0:
                self._accumulate_grad(out.grad.reshape(1))
                return
            grad_out = out.grad.reshape(out_shape)
            grad_in = np.broadcast_to(grad_out, self.shape).ravel()
            self._accumulate_grad(grad_in)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Public function mean."""
        summed_tensor = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = float(self.size)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else (axis,)
            axes = tuple(ax + self.ndim if ax < 0 else ax for ax in axes)
            count = float(prod(self.shape[ax] for ax in axes))
        return summed_tensor / count

    def reshape(self, new_shape):
        """Public function reshape."""
        if prod(new_shape) != self.size:
            raise ValueError("New shape must have the same number of elements as the original shape.")

        out = Tensor(self.data, new_shape, requires_grad=self.requires_grad, _children=(self,), _op="reshape")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def transpose(self):
        """Public function transpose."""
        if self.ndim != 2:
            raise ValueError("Transpose requires a 2D tensor.")

        rows, cols = self.shape
        arr = self.data.reshape(self.shape)
        out_data = arr.T.ravel()

        out = Tensor(out_data, (cols, rows), requires_grad=self.requires_grad, _children=(self,), _op="transpose")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_out = out.grad.reshape(cols, rows)
            self._accumulate_grad(grad_out.T.ravel())

        out._backward = _backward
        return out

    def _validate(self):
        if not isinstance(self.shape, tuple):
            raise ValueError("Shape must be a tuple.")
        if len(self.data) != self.size:
            raise ValueError(
                "Data size does not match the specified shape. "
                f"Expected {self.size} data values, got {len(self.data)} data values."
            )

    def flatten(self, start_dim=0, end_dim=-1):
        """Public function flatten."""
        if self.ndim == 0:
            if start_dim not in (0, -1) or end_dim not in (0, -1):
                raise ValueError("start_dim and end_dim must be 0 or -1 for scalar tensors.")
            return self.reshape((1,))

        if start_dim < 0:
            start_dim += self.ndim
        if end_dim < 0:
            end_dim += self.ndim

        if start_dim < 0 or end_dim < 0 or start_dim >= self.ndim or end_dim >= self.ndim:
            raise ValueError("start_dim and end_dim must be within tensor dimensions.")
        if start_dim > end_dim:
            raise ValueError("start_dim must be less than or equal to end_dim.")

        flattened = prod(self.shape[start_dim:end_dim + 1])
        new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
        return self.reshape(new_shape)
    
    def squeeze(self, axis=None):
        """Public function squeeze."""
        if axis is None:
            new_shape = tuple(dim for dim in self.shape if dim != 1)
            return self.reshape(new_shape if new_shape else ())
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        else:
            raise TypeError("axis must be int, tuple, list, or None")
        
        axes = tuple(ax + self.ndim if ax < 0 else ax for ax in axis)
        if len(set(axes)) != len(axes):
            raise ValueError("axis has duplicates")
        if any(ax < 0 or ax >= self.ndim for ax in axes):
            raise ValueError(f"axis out of range for ndim {self.ndim}")
        
        for ax in axes:
            if self.shape[ax] != 1:
                raise ValueError(f"Cannot squeeze dimension {ax} with size {self.shape[ax]}")
        
        new_shape = tuple(dim for i, dim in enumerate(self.shape) if i not in axes)
        return self.reshape(new_shape if new_shape else ())

    def unsqueeze(self, axis):
        """Public function unsqueeze."""
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        else:
            raise TypeError("axis must be int, tuple, or list")

        new_ndim = self.ndim + len(axis)
        axes = tuple(ax + new_ndim if ax < 0 else ax for ax in axis)
        if len(set(axes)) != len(axes):
            raise ValueError("axis has duplicates")
        if any(ax < 0 or ax >= new_ndim for ax in axes):
            raise ValueError(f"axis out of range for resulting ndim {new_ndim}")

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
