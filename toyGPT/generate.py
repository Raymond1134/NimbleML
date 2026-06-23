"""One-shot text generation from a toy GPT checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig
from toyGPT.inference import load_for_inference
from toyGPT.sampling import sample_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text from a toy GPT checkpoint.")
    parser.add_argument("--config", type=Path, default=TOYGPT_ROOT / "gpt_toy_config.toml")
    parser.add_argument("--checkpoint", type=str, default="best", help="best, latest, step_N, or path")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (empty = token 0)")
    parser.add_argument("--chars", type=int, default=0, help="Max new tokens (0 = config sample_chars)")
    parser.add_argument("--temperature", type=float, default=-1.0, help="-1 = config default")
    parser.add_argument("--top-p", type=float, default=-1.0, help="-1 = config default; 1.0 = off")
    parser.add_argument("--top-k", type=int, default=-1, help="-1 = config default; 0 = off")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=-1.0,
        help="GPT-2 repetition penalty (1.0 = off, -1 = config default)",
    )
    args = parser.parse_args(argv)

    cfg = ToyGPTConfig.from_toml(args.config.resolve())
    model, tokenizer, state, ckpt_dir = load_for_inference(cfg, args.checkpoint)

    prompt_ids = tokenizer.encode(args.prompt) if args.prompt.strip() else [0]
    temperature = cfg.temperature if args.temperature < 0 else args.temperature
    top_p = cfg.top_p if args.top_p < 0 else args.top_p
    top_k = cfg.top_k if args.top_k < 0 else args.top_k
    max_new = args.chars if args.chars > 0 else cfg.sample_chars
    rep_pen = cfg.repetition_penalty if args.repetition_penalty < 0 else args.repetition_penalty

    text = sample_text(
        model,
        tokenizer,
        prompt_ids=prompt_ids,
        seq_len=model.max_seq_len,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=rep_pen,
    )
    print(text)
    print(
        f"\n# ckpt={ckpt_dir.name} step={state.get('step')} temp={temperature} "
        f"top_p={top_p} top_k={top_k} rep_pen={rep_pen} len={max_new}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
