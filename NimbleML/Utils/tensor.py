# tensor.py
# Minimal autograd tensor for 1D/2D workloads.
from math import prod

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
        self.strides = self._compute_strides()

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

    # Only supports broadcasting for 1D/2D tensors. More complex broadcasting should be implemented later.
    def _apply_binary(self, other, op, grad_a_rule, grad_b_rule, op_name):
        other = self._ensure_tensor(other)
        a = self
        b = other

        if a.shape == b.shape:
            out_shape = a.shape
            out_data = [op(x, y) for x, y in zip(a.data, b.data)]
            case = "same"
        elif a.shape == ():
            out_shape = b.shape
            a_val = a.item()
            out_data = [op(a_val, y) for y in b.data]
            case = "a_scalar"
        elif b.shape == ():
            out_shape = a.shape
            b_val = b.item()
            out_data = [op(x, b_val) for x in a.data]
            case = "b_scalar"
        elif a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]:
            rows, cols = a.shape
            out_shape = a.shape
            out_data = []
            for i in range(rows):
                base = i * cols
                for j in range(cols):
                    out_data.append(op(a.data[base + j], b.data[j]))
            case = "a2_b1"
        elif a.ndim == 1 and b.ndim == 2 and b.shape[1] == a.shape[0]:
            rows, cols = b.shape
            out_shape = b.shape
            out_data = []
            for i in range(rows):
                base = i * cols
                for j in range(cols):
                    out_data.append(op(a.data[j], b.data[base + j]))
            case = "a1_b2"
        else:
            raise ValueError("Shapes are not compatible for this operation.")

        out = Tensor(out_data, out_shape, requires_grad=a.requires_grad or b.requires_grad, _children=(a, b), _op=op_name)

        def _backward():
            if out.grad is None:
                return
            grad_out = out.grad

            if case == "same":
                if a.requires_grad:
                    grad_a = [grad_a_rule(grad, x, y) for grad, x, y in zip(grad_out, a.data, b.data)]
                    a._accumulate_grad(grad_a)
                if b.requires_grad:
                    grad_b = [grad_b_rule(grad, x, y) for grad, x, y in zip(grad_out, a.data, b.data)]
                    b._accumulate_grad(grad_b)
                return

            if case == "a_scalar":
                if a.requires_grad:
                    a_val = a.item()
                    grad_a = [sum(grad_a_rule(grad, a_val, y) for grad, y in zip(grad_out, b.data))]
                    a._accumulate_grad(grad_a)
                if b.requires_grad:
                    a_val = a.item()
                    grad_b = [grad_b_rule(grad, a_val, y) for grad, y in zip(grad_out, b.data)]
                    b._accumulate_grad(grad_b)
                return

            if case == "b_scalar":
                if a.requires_grad:
                    b_val = b.item()
                    grad_a = [grad_a_rule(grad, x, b_val) for grad, x in zip(grad_out, a.data)]
                    a._accumulate_grad(grad_a)
                if b.requires_grad:
                    b_val = b.item()
                    grad_b = [sum(grad_b_rule(grad, x, b_val) for grad, x in zip(grad_out, a.data))]
                    b._accumulate_grad(grad_b)
                return

            if case == "a2_b1":
                rows, cols = a.shape
                if a.requires_grad:
                    grad_a = []
                    for i in range(rows):
                        base = i * cols
                        for j in range(cols):
                            idx = base + j
                            grad_a.append(grad_a_rule(grad_out[idx], a.data[idx], b.data[j]))
                    a._accumulate_grad(grad_a)
                if b.requires_grad:
                    grad_b = [0.0] * cols
                    for i in range(rows):
                        base = i * cols
                        for j in range(cols):
                            idx = base + j
                            grad_b[j] += grad_b_rule(grad_out[idx], a.data[idx], b.data[j])
                    b._accumulate_grad(grad_b)
                return

            if case == "a1_b2":
                rows, cols = b.shape
                if a.requires_grad:
                    grad_a = [0.0] * cols
                    for i in range(rows):
                        base = i * cols
                        for j in range(cols):
                            idx = base + j
                            grad_a[j] += grad_a_rule(grad_out[idx], a.data[j], b.data[idx])
                    a._accumulate_grad(grad_a)
                if b.requires_grad:
                    grad_b = []
                    for i in range(rows):
                        base = i * cols
                        for j in range(cols):
                            idx = base + j
                            grad_b.append(grad_b_rule(grad_out[idx], a.data[j], b.data[idx]))
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
        out_data = [0.0] * (rows * cols)

        for i in range(rows):
            for j in range(cols):
                acc = 0.0
                for k in range(inner):
                    acc += self.data[i * inner + k] * other.data[k * cols + j]
                out_data[i * cols + j] = acc

        out = Tensor(out_data, (rows, cols), requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="matmul")

        def _backward():
            if out.grad is None:
                return
            grad_out = out.grad

            if self.requires_grad:
                grad_self = [0.0] * (rows * inner)
                for i in range(rows):
                    for k in range(inner):
                        acc = 0.0
                        for j in range(cols):
                            acc += grad_out[i * cols + j] * other.data[k * cols + j]
                        grad_self[i * inner + k] = acc
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = [0.0] * (inner * cols)
                for k in range(inner):
                    for j in range(cols):
                        acc = 0.0
                        for i in range(rows):
                            acc += self.data[i * inner + k] * grad_out[i * cols + j]
                        grad_other[k * cols + j] = acc
                other._accumulate_grad(grad_other)

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

    def sum(self):
        out = Tensor([sum(self.data)], (), requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = [out.grad[0]] * self.size
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mean(self):
        return self.sum() / self.size

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

    def _compute_strides(self):
        strides = []
        current_stride = 1
        for dim in reversed(self.shape):
            strides.insert(0, current_stride)
            current_stride *= dim
        return tuple(strides)

    def _flat_index(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices)
        flat_index = 0
        for idx, stride in zip(indices, self.strides):
            flat_index += idx * stride
        return flat_index
