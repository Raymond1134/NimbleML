"""Shared helpers for layer implementations"""


def _kernel_dims(kernel_size):
    if isinstance(kernel_size, int):
        return kernel_size, kernel_size
    return kernel_size
