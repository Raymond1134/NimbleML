"""Axis index normalization for tensor reductions and reshapes."""
from __future__ import annotations


def normalize_axis(ndim: int, axis: int) -> int:
    """Convert a possibly-negative axis into a valid index in ``[0, ndim)``.

    Args:
        ndim: Number of dimensions in the tensor.
        axis: Axis index (may be negative).

    Returns:
        Non-negative axis in ``[0, ndim)``.

    Raises:
        ValueError: If ``ndim < 1`` or ``axis`` is out of bounds.
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got ndim={ndim}.")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for ndim={ndim}.")
    return axis


def normalize_axes(ndim: int, axis: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Normalize one or more axis indices.

    Args:
        ndim: Number of dimensions in the tensor.
        axis: Single axis or sequence of axes (may be negative).

    Returns:
        Tuple of non-negative axis indices.

    Raises:
        TypeError: If ``axis`` is not an int, tuple, or list.
        ValueError: If any axis is out of bounds.
    """
    if isinstance(axis, int):
        axes = (axis,)
    elif isinstance(axis, (list, tuple)):
        axes = tuple(axis)
    else:
        raise TypeError("axis must be int, tuple, or list")
    return tuple(normalize_axis(ndim, ax) for ax in axes)
