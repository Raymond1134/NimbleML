"""Shape helpers shared across layers."""


def kernel_dims(kernel_size):
    """Return (kH, kW) for an int or (kH, kW) kernel_size tuple."""
    if isinstance(kernel_size, int):
        return kernel_size, kernel_size
    return kernel_size
