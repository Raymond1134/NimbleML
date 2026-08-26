"""Load the required ``nimbleml_native`` extension.

NimbleML does not ship a Python fallback for hot kernels. Build the extension by doing::

    pip install -e ".[dev]"

Prerequisites: C++ toolchain (MSVC on Windows), CMake, and pybind11.
"""
from __future__ import annotations
import os

_INSTALL_HINT = (
    "NimbleML requires the compiled extension 'nimbleml_native'.\\n"
    "Install build tools, then from the repo root run:\\n"
    "  pip install -e \".[dev]\"\\n"
    "Windows: Visual Studio Build Tools (C++), CMake (pip install cmake).\\n"
    "Optional CUDA: set NIMBLEML_WITH_CUDA=ON in CMake for flash-SDPA."
)


def _add_windows_cuda_dlls() -> None:
    """CUDA-built wheels need ``cudart64_*.dll`` on the Windows DLL search path."""
    if os.name != "nt":
        return
    cuda = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda or not os.path.isdir(os.path.join(cuda, "bin")):
        toolkit = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA GPU Computing Toolkit",
            "CUDA",
        )
        if os.path.isdir(toolkit):
            for ver in sorted(
                (p for p in os.listdir(toolkit) if p.startswith("v")),
                reverse=True,
            ):
                candidate = os.path.join(toolkit, ver)
                if os.path.isdir(os.path.join(candidate, "bin")):
                    cuda = candidate
                    os.environ.setdefault("CUDA_PATH", cuda)
                    os.environ.setdefault("CUDA_HOME", cuda)
                    break
    if not cuda:
        return
    bindir = os.path.join(cuda, "bin")
    if not os.path.isdir(bindir):
        return
    path = os.environ.get("PATH", "")
    if bindir not in path.split(";"):
        os.environ["PATH"] = bindir + ";" + path
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        try:
            add(bindir)
        except OSError:
            pass


def load_native():
    _add_windows_cuda_dlls()
    try:
        import nimbleml_native as native
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return native


native = load_native()

__all__ = ["native", "load_native"]
