"""Run NimbleML benchmarks (GPU default)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.benchmark_ops import main

if __name__ == "__main__":
    raise SystemExit(main())
