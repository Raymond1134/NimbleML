"""Prepare the FineWeb-Edu token corpus (download + BPE train + encode -> bins).

Run this once up front; training/resume then memmap the cached bins instantly:

    python toyGPT\\prepare_data.py
    python toyGPT\\prepare_data.py --config toyGPT\\gpt_toy_config.toml

It is safe to re-run: completed tokenizer/bins are reused (cached by content
hash), so an interrupted prep simply continues from a clean rebuild.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig
from toyGPT.fineweb import prepare_corpus


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu tokens for toyGPT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=TOYGPT_ROOT / "gpt_toy_config.toml",
        help="Path to training config TOML.",
    )
    args = parser.parse_args(argv)

    cfg = ToyGPTConfig.from_toml(args.config.resolve())
    print(
        f"[prep] dataset={cfg.dataset}/{cfg.hf_subset} vocab={cfg.vocab_size} "
        f"train_tokens={cfg.train_tokens:,} val_tokens={cfg.val_tokens:,}"
    )
    t0 = time.perf_counter()
    tokenizer, tok_path, train_path, val_path, meta = prepare_corpus(cfg, verbose=True)
    print(
        f"[prep] done in {time.perf_counter() - t0:.1f}s | "
        f"train_tokens={meta['train_tokens']:,} val_tokens={meta['val_tokens']:,}"
    )
    print(f"[prep] tokenizer: {tok_path}")
    print(f"[prep] train bin: {train_path}")
    print(f"[prep] val bin:   {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
