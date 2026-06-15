# train_gpt.py
# Token-level (BPE) GPT training and generation sandbox
import math
import time
from pathlib import Path

import numpy as host_np

from NimbleML.data.text import batch_sequences, load_text_bpe
from NimbleML.losses import CrossEntropyLoss
from NimbleML.models.gpt import GPT
from NimbleML.optimizers import Adam
from NimbleML.utils.np_backend import device, np, using_gpu
from NimbleML.utils.tensor import Tensor

CORPUS_PATH = Path(__file__).parent / "NimbleML" / "data" / "tiny_corpus.txt"
TOKENIZER_PATH = Path(__file__).parent / "NimbleML" / "data" / "tokenizer_512.json"

VOCAB_SIZE = 512
TARGET_TRAIN_SECONDS = 600  # ~10 min experiment: big model, aggressive LR
BATCH_SIZE = 4  # smaller batch to fit wider/deeper model and keep step count reasonable
SEQ_LEN = 128
MAX_SEQ_LEN = SEQ_LEN
D_MODEL = 320
NUM_HEADS = 8
NUM_LAYERS = 10
FF_MULT = 4
LEARNING_RATE = 1e-3  # ~3x default; paired with warmup + no cosine decay
LR_WARMUP_STEPS = 40
BPE_TRAIN_CHARS = 2_000_000

USE_RECOMMENDED_SETTINGS = True
LOG_EVERY = 10

# Generation defaults tuned for readable, varied prose (not greedy repetition)
GENERATION_TOKENS = 300
TEMPERATURE = 0.85
TOP_K = 40

PLAYGROUND_HELP = """
Playground commands (prefix with /):
  /temp <float>   sampling temperature (0 = greedy argmax)
  /topk <int>     top-k filter (0 = off, try 40)
  /len <int>      max new tokens per generation
  /settings       show current generation settings
  /help           this message
  quit / exit     leave playground

Anything else is used as a generation seed prompt.
"""


def count_parameters(model):
    return sum(param.size for param in model.parameters())


def batches_in_corpus(num_tokens, batch_size, seq_len):
    return num_tokens // (batch_size * (seq_len + 1))


def cosine_lr(base_lr, step, total_steps):
    if total_steps <= 1:
        return base_lr
    progress = min(step / total_steps, 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def aggressive_lr(base_lr, step, warmup_steps):
    """Linear warmup then hold peak LR — favors fast early loss drop on short runs."""
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _prompt_value(label, default, cast):
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"Invalid value, keeping default {default}.")
        return default


def recommended_config(num_tokens):
    batches = batches_in_corpus(num_tokens, BATCH_SIZE, SEQ_LEN)
    # Bigger models are slower per step; use a conservative estimate for logging only.
    est_step_seconds = 0.45
    est_steps = max(1, int(TARGET_TRAIN_SECONDS / est_step_seconds))
    est_epochs = max(1, est_steps // max(batches, 1))
    return {
        "target_train_seconds": TARGET_TRAIN_SECONDS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "batches_per_epoch": batches,
        "estimated_epochs": est_epochs,
        "learning_rate": LEARNING_RATE,
        "lr_warmup_steps": LR_WARMUP_STEPS,
        "lr_schedule": "warmup + constant (aggressive)",
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ff_mult": FF_MULT,
        "max_seq_len": MAX_SEQ_LEN,
        "generation_tokens": GENERATION_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
    }


def configure_training(num_tokens):
    if USE_RECOMMENDED_SETTINGS:
        cfg = recommended_config(num_tokens)
        print("\nUsing recommended GPT settings (big model, ~10 min aggressive training):")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
        print()
        return cfg

    print("\nGPT training sandbox — press Enter to keep each default.\n")
    default_batches = batches_in_corpus(num_tokens, BATCH_SIZE, SEQ_LEN)
    return {
        "target_train_seconds": _prompt_value("Train seconds", TARGET_TRAIN_SECONDS, int),
        "batch_size": _prompt_value("Batch size", BATCH_SIZE, int),
        "seq_len": _prompt_value("Sequence length", SEQ_LEN, int),
        "batches_per_epoch": default_batches,
        "estimated_epochs": _prompt_value("Epochs (hint only)", 25, int),
        "learning_rate": _prompt_value("Learning rate", LEARNING_RATE, float),
        "d_model": _prompt_value("Model width (d_model)", D_MODEL, int),
        "num_heads": _prompt_value("Num heads", NUM_HEADS, int),
        "num_layers": _prompt_value("Num layers", NUM_LAYERS, int),
        "ff_mult": _prompt_value("FFN multiplier", FF_MULT, int),
        "max_seq_len": _prompt_value("Max seq len", MAX_SEQ_LEN, int),
        "generation_tokens": _prompt_value("Generation length", GENERATION_TOKENS, int),
        "temperature": _prompt_value("Temperature", TEMPERATURE, float),
        "top_k": _prompt_value("Top-k (0=off)", TOP_K, int),
    }


def train_step(model, criterion, optimizer, inputs, targets):
    logits = model(inputs)
    optimizer.zero_grad()
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return float(loss.data[0])


def _host_array(values):
    arr = np.asarray(values).ravel()
    if hasattr(arr, "get"):
        return arr.get()
    return host_np.asarray(arr)


def _sample_from_logits(logits, temperature, top_k=0):
    logits = _host_array(logits).astype(host_np.float64, copy=False)
    if top_k > 0:
        k = min(top_k, len(logits))
        keep = logits.argpartition(-k)[-k:]
        masked = host_np.full_like(logits, -host_np.inf)
        masked[keep] = logits[keep]
        logits = masked

    if temperature <= 0:
        return int(logits.argmax())
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    probs = host_np.exp(scaled)
    probs = probs / probs.sum()
    return int(host_np.random.choice(len(probs), p=probs))


def _context_ids(full_ids, max_seq_len):
    if len(full_ids) <= max_seq_len:
        return list(full_ids)
    return list(full_ids[-max_seq_len:])


def generate(model, prompt, tokenizer, max_seq_len, max_new_tokens, temperature, top_k):
    full_ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        window = _context_ids(full_ids, max_seq_len)
        inputs = Tensor(np.array(window, dtype=np.int64), (1, len(window)))
        logits = model(inputs)
        seq_len = logits.shape[1]
        last_logits = _host_array(logits.data.reshape(1, seq_len, -1)[0, -1])
        next_id = _sample_from_logits(last_logits, temperature, top_k)
        full_ids.append(next_id)
    return tokenizer.decode(full_ids)


def _format_gen_settings(cfg):
    return (
        f"temp={cfg['temperature']}, top_k={cfg['top_k'] or 'off'}, "
        f"max_new_tokens={cfg['generation_tokens']}, context={cfg['max_seq_len']}"
    )


def _handle_playground_command(line, cfg):
    cmd = line.strip().lower()
    if cmd in {"/help", "/h", "/?"}:
        print(PLAYGROUND_HELP)
        return True
    if cmd in {"/settings", "/cfg"}:
        print(_format_gen_settings(cfg))
        return True

    parts = line.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name == "/temp":
        if not arg:
            print(f"temp={cfg['temperature']}")
            return True
        try:
            cfg["temperature"] = float(arg)
            print(f"temp={cfg['temperature']}")
        except ValueError:
            print("Usage: /temp <float>  (e.g. /temp 0.85, /temp 0 for greedy)")
        return True

    if name == "/topk":
        if not arg:
            print(f"top_k={cfg['top_k']}")
            return True
        try:
            cfg["top_k"] = int(arg)
            print(f"top_k={cfg['top_k'] or 'off'}")
        except ValueError:
            print("Usage: /topk <int>  (e.g. /topk 40, /topk 0 to disable)")
        return True

    if name in {"/len", "/length", "/tokens"}:
        if not arg:
            print(f"generation_tokens={cfg['generation_tokens']}")
            return True
        try:
            value = int(arg)
            if value < 1:
                raise ValueError
            cfg["generation_tokens"] = value
            print(f"generation_tokens={cfg['generation_tokens']}")
        except ValueError:
            print("Usage: /len <int>  (e.g. /len 300)")
        return True

    print(f"Unknown command {parts[0]!r}. Type /help for options.")
    return True


def playground(model, tokenizer, cfg):
    print("\n--- GPT playground ---")
    print(_format_gen_settings(cfg))
    print("Type /help for commands, or enter a seed prompt. quit / exit to leave.\n")

    while True:
        seed = input("seed> ")
        stripped = seed.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in {"quit", "exit"}:
            print("Bye.")
            break
        if stripped.startswith("/"):
            _handle_playground_command(stripped, cfg)
            continue

        text = generate(
            model,
            seed,
            tokenizer,
            cfg["max_seq_len"],
            cfg["generation_tokens"],
            cfg["temperature"],
            cfg["top_k"],
        )
        print(f"\n{text}\n")


def main():
    ids, tokenizer = load_text_bpe(
        CORPUS_PATH,
        TOKENIZER_PATH,
        vocab_size=VOCAB_SIZE,
        max_train_chars=BPE_TRAIN_CHARS,
    )
    cfg = configure_training(len(ids))

    vocab_size = tokenizer.vocab_size
    batches_per_epoch = batches_in_corpus(len(ids), cfg["batch_size"], cfg["seq_len"])
    est_total_steps = max(1, int(cfg["target_train_seconds"] / 0.45))
    warmup_steps = cfg.get("lr_warmup_steps", LR_WARMUP_STEPS)

    print(f"Device: {device} (gpu={using_gpu})")
    print(f"Corpus: {CORPUS_PATH.name} | bpe vocab={vocab_size} | tokens={len(ids):,}")
    print(
        f"Training budget: {cfg['target_train_seconds']}s (~{cfg['target_train_seconds'] / 3600:.1f}h) | "
        f"~{batches_per_epoch} batches/epoch | est. ~{est_total_steps // max(batches_per_epoch, 1)} epochs\n"
    )

    model = GPT(
        vocab_size,
        cfg["d_model"],
        cfg["num_heads"],
        cfg["num_layers"],
        cfg["max_seq_len"],
        ff_mult=cfg["ff_mult"],
    )
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}\n")

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=cfg["learning_rate"])

    deadline = time.time() + cfg["target_train_seconds"]
    global_step = 0
    epoch = 0
    train_start = time.time()

    while time.time() < deadline:
        epoch += 1
        epoch_loss = 0.0
        steps = 0
        for inputs, targets in batch_sequences(ids, cfg["batch_size"], cfg["seq_len"]):
            if time.time() >= deadline:
                break

            global_step += 1
            optimizer.learning_rate = aggressive_lr(
                cfg["learning_rate"], global_step, warmup_steps
            )
            epoch_loss += train_step(model, criterion, optimizer, inputs, targets)
            steps += 1

            if LOG_EVERY > 0 and global_step % LOG_EVERY == 0:
                elapsed = time.time() - train_start
                remaining = max(0.0, deadline - time.time())
                print(
                    f"  step {global_step} | epoch {epoch} | batch {steps}/{batches_per_epoch} "
                    f"| loss {epoch_loss / steps:.4f} | lr {optimizer.learning_rate:.2e} "
                    f"| {elapsed / 60:.1f}m elapsed, {remaining / 60:.1f}m left"
                )

        if steps == 0:
            break
        avg_loss = epoch_loss / steps
        print(f"epoch {epoch} done | avg loss {avg_loss:.4f} | {steps} batches")

    elapsed = time.time() - train_start
    print(
        f"\nTraining finished: {global_step} steps, {epoch} epochs, "
        f"{elapsed / 60:.1f} minutes, final lr {optimizer.learning_rate:.2e}\n"
    )

    demo_seed = "The prince "
    sample = generate(
        model,
        demo_seed,
        tokenizer,
        cfg["max_seq_len"],
        min(120, cfg["generation_tokens"]),
        cfg["temperature"],
        cfg["top_k"],
    )
    print(f"Demo generation ({demo_seed!r}):\n{sample}\n")

    playground(model, tokenizer, cfg)


if __name__ == "__main__":
    main()
