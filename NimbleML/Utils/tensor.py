# tensor.py
# tensor class implementation in Python
from itertools import product
from math import prod

class Tensor:
    def __init__(self, data, shape=None):
        self.data = data
        self.shape = shape
        self._validate()
        self.strides = self._compute_strides()

    def __getitem__(self, indices):
        return self.data[self._flat_index(indices)]

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __div__(self, other):
        return self.__truediv__(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor.")

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D tensors.")

        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication.")

        rows, inner = self.shape
        _, cols = other.shape
        result = []

        for i in range(rows):
            for j in range(cols):
                acc = 0
                for k in range(inner):
                    acc += self.data[self._flat_index((i, k))] * other.data[other._flat_index((k, j))]
                result.append(acc)

        return Tensor(result, (rows, cols))

    def transpose(self):
        if self.ndim != 2:
            raise ValueError("Transpose requires a 2D tensor.")

        rows, cols = self.shape
        row_stride, col_stride = self.strides
        result = []

        for i in range(cols):
            base = i * col_stride
            for j in range(rows):
                result.append(self.data[j * row_stride + base])

        return Tensor(result, (cols, rows))

    def reshape(self, new_shape):
        if prod(new_shape) != self.size:
            raise ValueError("New shape must have the same number of elements as the original shape.")
        
        return Tensor(self.data, new_shape)
    
    def sum(self, axis=None):
        if axis is None:
            return sum(self.data)

        if axis < 0 or axis >= self.ndim:
            raise ValueError("Axis out of bounds.")

        new_shape = self.shape[:axis] + self.shape[axis+1:]
        result = [0] * prod(new_shape)

        for out_index in self._iter_indices(new_shape):
            in_index = list(out_index)
            in_index.insert(axis, 0)
            for i in range(self.shape[axis]):
                in_index[axis] = i
                result[self._flat_index(tuple(out_index))] += self.data[self._flat_index(tuple(in_index))]

        return Tensor(result, new_shape)
    
    def mean(self, axis=None):
        if axis is None:
            return sum(self.data) / self.size

        if axis < 0 or axis >= self.ndim:
            raise ValueError("Axis out of bounds.")

        new_shape = self.shape[:axis] + self.shape[axis+1:]
        result = [0] * prod(new_shape)

        for out_index in self._iter_indices(new_shape):
            in_index = list(out_index)
            in_index.insert(axis, 0)
            for i in range(self.shape[axis]):
                in_index[axis] = i
                result[self._flat_index(tuple(out_index))] += self.data[self._flat_index(tuple(in_index))]

        count = self.shape[axis]
        result = [x / count for x in result]
        return Tensor(result, new_shape)
    
    def max(self, axis=None):
        if axis is None:
            return max(self.data)

        if axis < 0 or axis >= self.ndim:
            raise ValueError("Axis out of bounds.")

        new_shape = self.shape[:axis] + self.shape[axis+1:]
        result = [float('-inf')] * prod(new_shape)

        for out_index in self._iter_indices(new_shape):
            in_index = list(out_index)
            in_index.insert(axis, 0)
            for i in range(self.shape[axis]):
                in_index[axis] = i
                value = self.data[self._flat_index(tuple(in_index))]
                result[self._flat_index(tuple(out_index))] = max(result[self._flat_index(tuple(out_index))], value)

        return Tensor(result, new_shape)

    @property
    def ndim(self): return len(self.shape)

    @property
    def size(self): return prod(self.shape)

    @property
    def T(self):
        return self.transpose()

    def _binary_op(self, other, op):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor.")

        out_shape = self._broadcast_shape(self.shape, other.shape)

        if out_shape == self.shape and out_shape == other.shape:
            result = [op(a, b) for a, b in zip(self.data, other.data)]
            return Tensor(result, out_shape)

        result = []
        for out_index in self._iter_indices(out_shape):
            left_index = self._broadcast_index(out_index, self.shape)
            right_index = self._broadcast_index(out_index, other.shape)
            left_value = self.data[self._flat_index(left_index)]
            right_value = other.data[other._flat_index(right_index)]
            result.append(op(left_value, right_value))

        return Tensor(result, out_shape)

    def _validate(self):
        if not isinstance(self.shape, tuple):
            raise ValueError("Shape must be a tuple.")

        if len(self.data) != self.size:
            raise ValueError(f"Data size does not match the specified shape. Expected {self.size} data values, got {len(self.data)} data values.")

    def _compute_strides(self):
        strides = []
        current_stride = 1

        for dim in reversed(self.shape):
            strides.insert(0, current_stride)
            current_stride *= dim

        return tuple(strides)

    def _flat_index(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        flat_index = 0
        for idx, stride in zip(indices, self.strides):
            flat_index += idx * stride

        return flat_index

    def _broadcast_shape(self, shape1, shape2):
        shape1 = self._pad_shape(shape1, max(len(shape1), len(shape2)))
        shape2 = self._pad_shape(shape2, max(len(shape1), len(shape2)))
        result = []

        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2:
                result.append(dim1)
            elif dim1 == 1:
                result.append(dim2)
            elif dim2 == 1:
                result.append(dim1)
            else:
                raise ValueError("Shapes cannot be broadcast together.")

        return tuple(result)

    def _pad_shape(self, shape, length):
        return (1,) * (length - len(shape)) + shape

    def _broadcast_index(self, out_index, in_shape):
        if len(in_shape) == 0:
            return ()

        offset = len(out_index) - len(in_shape)
        in_index = []

        for i, dim in enumerate(in_shape):
            out_dim = out_index[i + offset]
            in_index.append(0 if dim == 1 else out_dim)

        return tuple(in_index)

    def _iter_indices(self, shape):
        if len(shape) == 0:
            return ()

        for index in product(*[range(dim) for dim in shape]):
            yield index
