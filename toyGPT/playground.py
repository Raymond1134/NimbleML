"""Interactive Nova chatbot — sample from a trained toy GPT checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toyGPT.config import TOYGPT_ROOT, ToyGPTConfig
from toyGPT.inference import load_for_inference
from toyGPT.sampling import sample_text

ASSISTANT_NAME = "Nova"
USER_NAME = "You"

HELP = f"""
Chat commands (leading ':'):
  :help              show this help
  :clear             start a new conversation
  :temp <float>      sampling temperature (0 = greedy)
  :topk <int>        top-k filter (0 = off)
  :len <int>         max new tokens per reply
  :ckpt best|latest|step_N   reload checkpoint
  :quit              exit

Anything else is sent to {ASSISTANT_NAME} as your message.
"""


@dataclass
class PlaySettings:
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 200
    checkpoint: str = "best"


@dataclass
class ChatSession:
    """Rolling transcript formatted as User / Nova turns."""

    turns: list[tuple[str, str]] = field(default_factory=list)

    def add_user(self, message: str) -> None:
        self.turns.append(("user", message.strip()))

    def add_assistant(self, message: str) -> None:
        self.turns.append(("assistant", message.strip()))

    def clear(self) -> None:
        self.turns.clear()

    def prompt_text(self) -> str:
        lines: list[str] = []
        for role, message in self.turns:
            speaker = USER_NAME if role == "user" else ASSISTANT_NAME
            lines.append(f"{speaker}: {message}")
        lines.append(f"{ASSISTANT_NAME}:")
        return "\n".join(lines)


def _trim_reply(text: str) -> str:
    """Stop generation if the model starts a new user turn."""
    for stop in (f"\n{USER_NAME}:", f"\n{USER_NAME} :", "\nUser:", "\nuser:"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def _fit_prompt_ids(tokenizer, prompt: str, max_len: int) -> list[int]:
    ids = tokenizer.encode(prompt)
    if len(ids) <= max_len:
        return ids
    return ids[-max_len:]


def _generate_reply(model, tokenizer, settings: PlaySettings, session: ChatSession) -> str:
    prompt = session.prompt_text()
    prompt_ids = _fit_prompt_ids(tokenizer, prompt, model.max_seq_len)
    raw = sample_text(
        model,
        tokenizer,
        prompt_ids=prompt_ids,
        seq_len=model.max_seq_len,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
        include_prompt=False,
    )
    return _trim_reply(raw)


def _print_banner(step: int | None, ckpt_name: str, settings: PlaySettings) -> None:
    step_label = f"step {step}" if step is not None else "unknown step"
    print(
        f"\n{ASSISTANT_NAME}: Hi — I'm {ASSISTANT_NAME}, your toy GPT assistant "
        f"({ckpt_name}, {step_label}).\n"
        f"temp={settings.temperature} top_k={settings.top_k} "
        f"max_tokens={settings.max_new_tokens}\n"
        f"Type :help for commands.\n"
    )


def _run_turn(model, tokenizer, settings: PlaySettings, session: ChatSession, user_message: str) -> None:
    session.add_user(user_message)
    print(f"\n{USER_NAME}: {user_message}\n")
    try:
        reply = _generate_reply(model, tokenizer, settings, session)
    except Exception as exc:
        session.turns.pop()
        print(f"[error] generation failed: {exc}\n")
        return

    if not reply:
        reply = "(no reply — try a longer prompt, lower temperature, or a later checkpoint)"
    session.add_assistant(reply)
    print(f"{ASSISTANT_NAME}: {reply}\n")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=f"Chat with {ASSISTANT_NAME} (toy GPT).")
    parser.add_argument("--config", type=Path, default=TOYGPT_ROOT / "gpt_toy_config.toml")
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--len", type=int, default=0, dest="gen_len")
    args = parser.parse_args(argv)

    cfg = ToyGPTConfig.from_toml(args.config.resolve())
    settings = PlaySettings(
        max_new_tokens=args.gen_len if args.gen_len > 0 else cfg.sample_chars,
        checkpoint=args.checkpoint,
    )
    if args.temperature >= 0:
        settings.temperature = args.temperature
    if args.top_k >= 0:
        settings.top_k = args.top_k

    model, tokenizer, state, ckpt_dir = load_for_inference(cfg, settings.checkpoint)
    session = ChatSession()
    _print_banner(state.get("step"), ckpt_dir.name, settings)

    while True:
        try:
            line = input(f"{USER_NAME}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{ASSISTANT_NAME}: Bye!\n")
            break

        if not line:
            continue
        if not line.startswith(":"):
            _run_turn(model, tokenizer, settings, session, line)
            continue

        parts = line[1:].split()
        cmd = parts[0].lower() if parts else "help"

        if cmd in ("quit", "q", "exit"):
            print(f"{ASSISTANT_NAME}: Bye!\n")
            break
        if cmd == "help":
            print(HELP.strip())
        elif cmd == "clear":
            session.clear()
            print(f"{ASSISTANT_NAME}: Fresh start — what would you like to talk about?\n")
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
            session.clear()
            _print_banner(state.get("step"), ckpt_dir.name, settings)
        else:
            print("Unknown command. Type :help")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
