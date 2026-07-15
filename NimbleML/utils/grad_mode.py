"""Gradient-mode helpers (``no_grad`` / ``enable_grad``)."""
from __future__ import annotations
from contextlib import contextmanager

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


@contextmanager
def no_grad():
    """Disable gradient tracking in the enclosed block."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


@contextmanager
def enable_grad():
    """Force-enable gradients in the enclosed block."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = prev
