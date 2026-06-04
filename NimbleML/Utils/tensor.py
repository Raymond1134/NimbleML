# tensor.py
# Minimal autograd tensor for 1D/2D workloads.
from math import prod
import numpy as np

class Tensor:
    def __init__(self, data, shape, requires_grad=False, _children=(), _op=""):
        if isinstance(data, (int, float)): data = [float(data)]
        self.data = list(data)
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._validate()
        self.strides = Tensor._compute_strides(self.shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __getitem__(self, indices):
        return self.data[self._flat_index(indices)]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return prod(self.shape) if self.shape else 1

    @property
    def T(self):
        return self.transpose()
    
    def item(self):
        if self.size != 1:
            raise ValueError("Only scalar tensors can be converted to a Python scalar.")
        return self.data[0]

    def zero_grad(self):
        self.grad = [0.0] * self.size

    def backward(self, grad=None):
        if grad is None:
            if self.size != 1:
                raise ValueError("grad must be specified for non-scalar tensors.")
            grad = [1.0]
        elif isinstance(grad, (int, float)):
            grad = [float(grad)]
        elif isinstance(grad, Tensor):
            grad = grad.data

        self._accumulate_grad(grad)

        topo = []
        visited = set()

        def build(node):
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
        if self.grad is None:
            self.grad = list(grad)
        else:
            self.grad = [grad + dgrad for grad, dgrad in zip(self.grad, grad)]

    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor([other], (), requires_grad=False)

    def __add__(self, other):
        return self._apply_binary(
            other,
            lambda a, b: a + b,
            lambda grad, a, b: grad,
            lambda grad, a, b: grad,
            "add",
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_binary(
            other,
            lambda a, b: a - b,
            lambda grad, a, b: grad,
            lambda grad, a, b: -grad,
            "sub",
        )

    def __rsub__(self, other):
        return self._ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        return self._apply_binary(
            other,
            lambda a, b: a * b,
            lambda grad, a, b: grad * b,
            lambda grad, a, b: grad * a,
            "mul",
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_binary(
            other,
            lambda a, b: a / b,
            lambda grad, a, b: grad / b,
            lambda grad, a, b: -(grad*a) / (b*b),
            "div",
        )

    def __rtruediv__(self, other):
        return self._ensure_tensor(other).__truediv__(self)

    def __matmul__(self, other):
        return self.matmul(other)

    def _apply_binary(self, other, op, grad_a_rule, grad_b_rule, op_name):
        other = self._ensure_tensor(other)
        a = self
        b = other
        out_shape, shape_a, shape_b = self._broadcast_shape(a.shape, b.shape)
        out_data = []
        
        for out_idx in Tensor._iter_indices(out_shape):
            a_idx_aligned = Tensor._broadcast_index(out_idx, shape_a)
            b_idx_aligned = Tensor._broadcast_index(out_idx, shape_b)
            a_idx = a_idx_aligned[-len(a.shape):] if a.shape else ()
            b_idx = b_idx_aligned[-len(b.shape):] if b.shape else ()
            out_data.append(op(a.data[a._flat_index(a_idx)], b.data[b._flat_index(b_idx)]))
        
        out = Tensor(out_data, out_shape, requires_grad=a.requires_grad or b.requires_grad, _children=(a, b), _op=op_name)
        
        def _backward():
            if out.grad is None:
                return
            grad_out = out.grad
            
            grad_a = [0.0] * a.size
            grad_b = [0.0] * b.size
            
            for out_idx in Tensor._iter_indices(out_shape):
                a_idx_aligned = Tensor._broadcast_index(out_idx, shape_a)
                b_idx_aligned = Tensor._broadcast_index(out_idx, shape_b)
                a_idx = a_idx_aligned[-len(a.shape):] if a.shape else ()
                b_idx = b_idx_aligned[-len(b.shape):] if b.shape else ()    
                out_flat = out._flat_index(out_idx)
                a_flat = a._flat_index(a_idx)
                b_flat = b._flat_index(b_idx)
                
                grad = grad_out[out_flat]
                grad_a[a_flat] += grad_a_rule(grad, a.data[a_flat], b.data[b_flat])
                grad_b[b_flat] += grad_b_rule(grad, a.data[a_flat], b.data[b_flat])

            if a.requires_grad:
                a._accumulate_grad(grad_a)
            if b.requires_grad:
                b._accumulate_grad(grad_b)

        out._backward = _backward
        return out
    
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

    @staticmethod
    def _iter_indices(shape):
        if len(shape) == 0:
            yield ()
            return;
        
        def iter(prefix, dim):
            if dim == len(shape):
                yield tuple(prefix)
                return
            for i in range(shape[dim]):
                prefix.append(i)
                yield from iter(prefix, dim + 1)
                prefix.pop()
        
        yield from iter([], 0)
    
    @staticmethod
    def _broadcast_index(out_idx, in_shape):
        indices = []
        for axis, dim in enumerate(in_shape):
            if dim == 1:
                indices.append(0)
            else:
                indices.append(out_idx[axis])
        return tuple(indices)

    def matmul(self, other):
        other = self._ensure_tensor(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D tensors.")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication.")

        rows, inner = self.shape
        _, cols = other.shape
        left = np.array(self.data, dtype=float).reshape(self.shape)
        right = np.array(other.data, dtype=float).reshape(other.shape)
        out_data = (left @ right).reshape(-1).tolist()

        out = Tensor(out_data, (rows, cols), requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="matmul")

        def _backward():
            if out.grad is None:
                return
            grad_out = out.grad

            grad_out_arr = np.array(grad_out, dtype=float).reshape(rows, cols)
            if self.requires_grad:
                other_arr = np.array(other.data, dtype=float).reshape(other.shape)
                grad_self = grad_out_arr @ other_arr.T
                self._accumulate_grad(grad_self.reshape(-1).tolist())
            if other.requires_grad:
                self_arr = np.array(self.data, dtype=float).reshape(self.shape)
                grad_other = self_arr.T @ grad_out_arr
                other._accumulate_grad(grad_other.reshape(-1).tolist())

        out._backward = _backward
        return out

    def relu(self):
        out_data = [val if val > 0 else 0.0 for val in self.data]
        out = Tensor(out_data, self.shape, requires_grad=self.requires_grad, _children=(self,), _op="relu")
        mask = [1.0 if val > 0 else 0.0 for val in self.data]

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = [g * m for g, m in zip(out.grad, mask)]
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            reduce_axes = set(range(self.ndim))
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
            reduce_axes = set(axis)

        if self.ndim == 0:
            out_shape = ()
        else:
            out_shape_list = []
            for i, dim in enumerate(self.shape):
                if i in reduce_axes:
                    if keepdims:
                        out_shape_list.append(1)
                else:
                    out_shape_list.append(dim)
            out_shape = tuple(out_shape_list) if out_shape_list else ()

        out_size = prod(out_shape) if out_shape else 1
        out_data = [0.0] * out_size
        out_strides = Tensor._compute_strides(out_shape)

        for in_idx in Tensor._iter_indices(self.shape):
            in_flat = self._flat_index(in_idx)
            if self.ndim == 0:
                out_idx = ()
            else:
                if keepdims:
                    out_idx = tuple(0 if i in reduce_axes else in_idx[i] for i in range(self.ndim))
                else:
                    out_idx = tuple(in_idx[i] for i in range(self.ndim) if i not in reduce_axes)
            out_flat = Tensor._flat_index_from(out_idx, out_strides)
            out_data[out_flat] += self.data[in_flat]

        out = Tensor(out_data, out_shape, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad_in = [0.0] * self.size
            for in_idx in Tensor._iter_indices(self.shape):
                in_flat = self._flat_index(in_idx)
                if self.ndim == 0:
                    out_idx = ()
                else:
                    if keepdims:
                        out_idx = tuple(0 if i in reduce_axes else in_idx[i] for i in range(self.ndim))
                    else:
                        out_idx = tuple(in_idx[i] for i in range(self.ndim) if i not in reduce_axes)
                out_flat = Tensor._flat_index_from(out_idx, out_strides)
                grad_in[in_flat] += out.grad[out_flat]
            self._accumulate_grad(grad_in)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        summed_tensor = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = float(self.size)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else (axis,)
            axes = tuple(ax + self.ndim if ax < 0 else ax for ax in axes)
            count = float(prod(self.shape[ax] for ax in axes))
        return summed_tensor / count

    def reshape(self, new_shape):
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
        if self.ndim != 2:
            raise ValueError("Transpose requires a 2D tensor.")

        rows, cols = self.shape
        out_data = [0.0] * (rows * cols)
        for i in range(rows):
            for j in range(cols):
                out_data[j * rows + i] = self.data[i * cols + j]

        out = Tensor(out_data, (cols, rows), requires_grad=self.requires_grad, _children=(self,), _op="transpose")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = [0.0] * (rows * cols)
            for i in range(rows):
                for j in range(cols):
                    grad[i * cols + j] = out.grad[j * rows + i]
            self._accumulate_grad(grad)

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
    
    @staticmethod
    def _compute_strides(shape):
        strides = []
        current_stride = 1
        for dim in reversed(shape):
            strides.insert(0, current_stride)
            current_stride *= dim
        return tuple(strides)

    @staticmethod
    def _flat_index_from(indices, strides):
        if not indices:
            return 0
        if not isinstance(indices, tuple):
            indices = (indices,)
        return sum(idx * stride for idx, stride in zip(indices, strides))

    def _flat_index(self, indices):
        return Tensor._flat_index_from(indices, self.strides)
