"""Load the required ``nimbleml_native`` extension.

NimbleML does not ship a Python fallback for hot kernels. Build the extension::

    pip install -e ".[dev]"

Prerequisites: C++ toolchain (MSVC on Windows), CMake, and pybind11.
"""
from __future__ import annotations

_INSTALL_HINT = (
    "NimbleML requires the compiled extension 'nimbleml_native'.\\n"
    "Install build tools, then from the repo root run:\\n"
    "  pip install -e \".[dev]\"\\n"
    "Windows: Visual Studio Build Tools (C++), CMake (pip install cmake).\\n"
    "Optional CUDA: set NIMBLEML_WITH_CUDA=ON in CMake for flash-SDPA."
)


def load_native():
    try:
        import nimbleml_native as native
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return native


native = load_native()

__all__ = ["native", "load_native"]
