"""
Gradient checker: compare autograd gradients to finite-difference estimates.

Gradcheck may report failures even when backward is correct:
- ReLU at x=0 (derivative is undefined; numeric and analytic can disagree)
- Floating-point noise (use tol around 1e-3, not 1e-8)
- Random layers like Dropout (use eval mode or a fixed seed)
- MaxPool ties (two equal values in a window — subgradient ambiguity)
"""
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def _randn(shape):
    if isinstance(shape, int):
        shape = (shape,)
    return np.random.randn(*shape).astype(np.float64)


def _scalar_value(value):
    if isinstance(value, Tensor):
        return value.item()
    return float(value)


def numerical_grad(fn, tensor, index=0, eps=1e-5):
    original = float(np.asarray(tensor.data, dtype=np.float64).ravel()[index])

    tensor.data[index] = original + eps
    plus = _scalar_value(fn())

    tensor.data[index] = original - eps
    minus = _scalar_value(fn())

    tensor.data[index] = original
    return (plus - minus) / (2 * eps)


def gradcheck(fn, tensors, eps=1e-4, tol=1e-3):
    if not tensors:
        raise ValueError("gradcheck requires at least one tensor.")

    for tensor in tensors:
        if not tensor.requires_grad:
            raise ValueError("All tensors passed to gradcheck must have requires_grad=True.")

    for tensor in tensors:
        tensor.grad = None

    loss = fn()
    if not isinstance(loss, Tensor) or loss.size != 1:
        raise ValueError("fn must return a scalar Tensor.")

    loss.backward()

    analytics = []
    for tensor in tensors:
        if tensor.grad is None:
            raise AssertionError(f"Missing analytic grad for tensor with shape {tensor.shape}.")
        analytics.append(np.asarray(tensor.grad, dtype=np.float64).ravel().copy())

    for tensor, analytic in zip(tensors, analytics):
        numeric = np.zeros(tensor.size, dtype=np.float64)

        for index in range(tensor.size):
            def scalar_fn():
                for t in tensors:
                    t.grad = None
                return _scalar_value(fn())

            numeric[index] = numerical_grad(scalar_fn, tensor, index=index, eps=eps)

        if not np.allclose(analytic, numeric, atol=tol, rtol=tol):
            max_diff = float(np.max(np.abs(analytic - numeric)))
            raise AssertionError(
                f"Gradcheck failed for tensor shape {tensor.shape}. "
                f"Max |analytic - numeric| = {max_diff:.6e} (tol={tol})."
            )

    return True
