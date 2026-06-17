"""Interactive sampling playground for a trained toy GPT checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig
from toyGPT.inference import load_for_inference
from toyGPT.sampling import sample_text

HELP = """
Playground commands (leading ':'):
  :help              show this help
  :temp <float>      sampling temperature (0 = greedy)
  :topk <int>        top-k filter (0 = off)
  :len <int>         max new tokens per generation
  :ckpt best|latest|step_N   reload checkpoint
  :quit              exit

Type any other line as a prompt and press Enter to generate.
"""


@dataclass
class PlaySettings:
    temperature: float = 0.8
    top_k: int = 0
    max_new_tokens: int = 200
    checkpoint: str = "best"


def _run_once(model, tokenizer, settings: PlaySettings, prompt: str) -> None:
    prompt_ids = tokenizer.encode(prompt) if prompt.strip() else [0]
    text = sample_text(
        model,
        tokenizer,
        prompt_ids=prompt_ids,
        seq_len=model.max_seq_len,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
    )
    print("\n---\n")
    print(text)
    print(
        f"\n--- temp={settings.temperature} top_k={settings.top_k} "
        f"len={settings.max_new_tokens} ckpt={settings.checkpoint} ---\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive toy GPT playground.")
    parser.add_argument("--config", type=Path, default=TOYGPT_ROOT / "gpt_toy_config.toml")
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--len", type=int, default=0, dest="gen_len")
    args = parser.parse_args(argv)

    cfg = ToyGPTConfig.from_toml(args.config.resolve())
    settings = PlaySettings(
        temperature=cfg.temperature if args.temperature < 0 else args.temperature,
        top_k=0 if args.top_k < 0 else args.top_k,
        max_new_tokens=args.gen_len if args.gen_len > 0 else cfg.sample_chars,
        checkpoint=args.checkpoint,
    )

    model, tokenizer, state, ckpt_dir = load_for_inference(cfg, settings.checkpoint)
    print(f"Loaded {ckpt_dir} (step {state.get('step')})")
    print(HELP.strip())

    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if not line.startswith(":"):
            _run_once(model, tokenizer, settings, line)
            continue

        parts = line[1:].split()
        cmd = parts[0].lower() if parts else "help"

        if cmd in ("quit", "q", "exit"):
            break
        if cmd == "help":
            print(HELP.strip())
        elif cmd == "temp" and len(parts) >= 2:
            settings.temperature = float(parts[1])
            print(f"temperature={settings.temperature}")
        elif cmd == "topk" and len(parts) >= 2:
            settings.top_k = int(parts[1])
            print(f"top_k={settings.top_k}")
        elif cmd == "len" and len(parts) >= 2:
            settings.max_new_tokens = int(parts[1])
            print(f"max_new_tokens={settings.max_new_tokens}")
        elif cmd == "ckpt" and len(parts) >= 2:
            settings.checkpoint = parts[1]
            model, tokenizer, state, ckpt_dir = load_for_inference(cfg, settings.checkpoint)
            print(f"reloaded {ckpt_dir} (step {state.get('step')})")
        else:
            print("Unknown command. Type :help")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
