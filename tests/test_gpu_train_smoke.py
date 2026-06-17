"""GPU training smoke test (runs in an isolated subprocess)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "_gpu_smoke_runner.py"


def _gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")
def test_gpu_train_20_steps_full_model():
    """20 training steps on the production 18L model in a fresh Python process."""
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"GPU smoke failed (code {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK:" in result.stdout
