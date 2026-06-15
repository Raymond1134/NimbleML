# gpt_common.py
# Shared helpers for train_gpt.py and playground.py (not part of the NimbleML library API).
import json
import math
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as host_np

from NimbleML.data.text import batch_sequences, load_text_bpe
from NimbleML.losses import CrossEntropyLoss
from NimbleML.models.gpt import GPT
from NimbleML.optimizers import Adam
from NimbleML.utils.np_backend import device, np, using_gpu
from NimbleML.utils.saveload import load as load_weights, named_parameters, save as save_weights
from NimbleML.utils.tensor import Tensor

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "NimbleML" / "data"
CHECKPOINT_DIR = ROOT / "checkpoints" / "latest"

CORPUS_PATH = DATA_DIR / "corpus.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer_4096.json"
IDS_CACHE_PATH = DATA_DIR / "corpus_ids_4096.npy"

# Salesforce S3 URLs are dead (301/403). This HF mirror hosts the original zip.
WIKITEXT103_ZIP_URL = (
    "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/"
    "wikitext-103-raw-v1.zip"
)

# ~50M param GPT — multi-day on a laptop GPU, much faster than the 227M config.
VOCAB_SIZE = 4096
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 14
SEQ_LEN = 256
MAX_SEQ_LEN = SEQ_LEN
FF_MULT = 4
BATCH_SIZE = 4

LEARNING_RATE = 6e-4
MIN_LR_RATIO = 0.05
WARMUP_STEPS = 500
COSINE_RESTART_EVERY = 5_000
GRAD_CLIP_NORM = 0.5
# ~3.8k merges; 1.5M chars is enough signal for a 4k vocab without hours of BPE training.
BPE_TRAIN_CHARS = 1_500_000

SAVE_EVERY_MINUTES = 10
LOG_EVERY = 5

GENERATION_TOKENS = 400
TEMPERATURE = 0.9
TOP_K = 50

PLAYGROUND_HELP = """
Playground commands (prefix with /):
  /temp <float>   sampling temperature (0 = greedy argmax)
  /topk <int>     top-k filter (0 = off, try 50)
  /len <int>      max new tokens per generation
  /settings       show current generation settings
  /help           this message
  quit / exit     leave playground

Anything else is used as a generation seed prompt.
"""


def count_parameters(model):
    return sum(param.size for param in model.parameters())


def batches_per_epoch(num_tokens, batch_size, seq_len):
    row_len = seq_len + 1
    return num_tokens // (batch_size * row_len)


def model_config(vocab_size=None):
    vocab_size = vocab_size or VOCAB_SIZE
    return {
        "vocab_size": vocab_size,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
        "ff_mult": FF_MULT,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "min_lr_ratio": MIN_LR_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "cosine_restart_every": COSINE_RESTART_EVERY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "bpe_train_chars": BPE_TRAIN_CHARS,
        "corpus_path": str(CORPUS_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
    }


def build_model(cfg):
    return GPT(
        cfg["vocab_size"],
        cfg["d_model"],
        cfg["num_heads"],
        cfg["num_layers"],
        cfg["max_seq_len"],
        ff_mult=cfg["ff_mult"],
    )


def _download_file(url, dest, chunk_size=1 << 20):
    """Download a URL to disk with redirect handling and basic progress."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "NimbleML-corpus-download/1.0"})
    try:
        response = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed ({exc.code}): {url}") from exc

    total = response.headers.get("Content-Length")
    total = int(total) if total else None
    downloaded = 0
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    with open(tmp_path, "wb") as out:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                print(f"\r  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)", end="")
            else:
                print(f"\r  {dest.name}: {downloaded / 1e6:.1f} MB", end="")

    print()
    tmp_path.replace(dest)
    return dest


def _find_token_file(root, split_name):
    """Locate wiki.{train,valid}.{tokens,raw} under an extracted WikiText tree."""
    for ext in ("tokens", "raw"):
        filename = f"wiki.{split_name}.{ext}"
        direct = root / filename
        if direct.exists():
            return direct
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find wiki.{split_name}.tokens or wiki.{split_name}.raw under {root}"
    )


def download_wikitext103():
    """Fetch WikiText-103 (Wikipedia prose) and build corpus.txt."""
    if CORPUS_PATH.exists() and CORPUS_PATH.stat().st_size > 1_000_000:
        print(f"Corpus already present: {CORPUS_PATH} ({CORPUS_PATH.stat().st_size / 1e6:.1f} MB)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "wikitext-103-raw-v1.zip"
    has_splits = any(DATA_DIR.rglob("wiki.train.tokens")) or any(DATA_DIR.rglob("wiki.train.raw"))

    if not has_splits:
        if not zip_path.exists():
            print("Downloading WikiText-103 (~192 MB) from Hugging Face mirror...")
            print(f"  {WIKITEXT103_ZIP_URL}")
            _download_file(WIKITEXT103_ZIP_URL, zip_path)
        print("Extracting WikiText-103...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)

    train_path = _find_token_file(DATA_DIR, "train")
    valid_path = _find_token_file(DATA_DIR, "valid")

    print("Building combined corpus (train + valid)...")
    parts = []
    for path in (train_path, valid_path):
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    corpus = "\n\n".join(parts)
    CORPUS_PATH.write_text(corpus, encoding="utf-8")
    print(f"Wrote {CORPUS_PATH} ({len(corpus):,} chars, {CORPUS_PATH.stat().st_size / 1e6:.1f} MB)")


def load_corpus_ids(verbose=True):
    """Tokenize corpus with caching so restarts skip the slow encode pass."""
    download_wikitext103()

    if IDS_CACHE_PATH.exists():
        if verbose:
            print(f"Loading cached token ids from {IDS_CACHE_PATH}...")
        ids = host_np.load(IDS_CACHE_PATH)
        from NimbleML.data.tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        if verbose:
            print(f"  {len(ids):,} tokens | vocab={tokenizer.vocab_size}")
        return ids, tokenizer

    ids, tokenizer = load_text_bpe(
        CORPUS_PATH,
        TOKENIZER_PATH,
        vocab_size=VOCAB_SIZE,
        max_train_chars=BPE_TRAIN_CHARS,
        verbose=verbose,
    )
    if verbose:
        print(f"Caching token ids to {IDS_CACHE_PATH}...")
    host_np.save(IDS_CACHE_PATH, host_np.asarray(ids, dtype=host_np.int32))
    return ids, tokenizer


def learning_rate_at_step(step, cfg):
    """Warmup + cosine annealing with periodic restarts (helps escape plateaus)."""
    base = cfg["learning_rate"]
    warmup = cfg["warmup_steps"]
    min_lr = base * cfg["min_lr_ratio"]
    restart = cfg["cosine_restart_every"]

    if step < warmup:
        return base * (step + 1) / max(warmup, 1)

    t = step - warmup
    cycle_pos = t % restart
    progress = cycle_pos / max(restart, 1)
    cycle_floor = min_lr + (base - min_lr) * 0.25 * (0.5 ** (t // restart))
    return cycle_floor + 0.5 * (base - cycle_floor) * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm_(params, max_norm):
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        g = param.grad
        if hasattr(g, "get"):
            g = g.get()
        total += float(host_np.sum(host_np.asarray(g, dtype=host_np.float64) ** 2))
    total_norm = math.sqrt(total)
    if total_norm <= max_norm or total_norm == 0.0:
        return total_norm

    scale = max_norm / total_norm
    for param in params:
        if param.grad is not None:
            param.grad = np.asarray(param.grad, dtype=param.grad.dtype) * scale
    return total_norm


def train_step(model, criterion, optimizer, inputs, targets, grad_clip):
    logits = model(inputs)
    optimizer.zero_grad()
    loss = criterion(logits, targets)
    loss.backward()
    if grad_clip > 0:
        clip_grad_norm_(model.parameters(), grad_clip)
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
        inputs = Tensor(host_np.array(window, dtype=host_np.int64), (1, len(window)))
        logits = model(inputs)
        seq_len = logits.shape[1]
        last_logits = _host_array(logits.data.reshape(1, seq_len, -1)[0, -1])
        next_id = _sample_from_logits(last_logits, temperature, top_k)
        full_ids.append(next_id)
    return tokenizer.decode(full_ids)


def generation_settings(cfg=None):
    cfg = cfg or model_config()
    return {
        "max_seq_len": cfg["max_seq_len"],
        "generation_tokens": GENERATION_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
    }


def format_gen_settings(cfg):
    return (
        f"temp={cfg['temperature']}, top_k={cfg['top_k'] or 'off'}, "
        f"max_new_tokens={cfg['generation_tokens']}, context={cfg['max_seq_len']}"
    )


def handle_playground_command(line, cfg):
    cmd = line.strip().lower()
    if cmd in {"/help", "/h", "/?"}:
        print(PLAYGROUND_HELP)
        return True
    if cmd in {"/settings", "/cfg"}:
        print(format_gen_settings(cfg))
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
            print("Usage: /temp <float>")
        return True

    if name == "/topk":
        if not arg:
            print(f"top_k={cfg['top_k']}")
            return True
        try:
            cfg["top_k"] = int(arg)
            print(f"top_k={cfg['top_k'] or 'off'}")
        except ValueError:
            print("Usage: /topk <int>")
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
            print("Usage: /len <int>")
        return True

    print(f"Unknown command {parts[0]!r}. Type /help for options.")
    return True


def run_playground(model, tokenizer, cfg):
    gen_cfg = generation_settings(cfg)
    print("\n--- GPT playground ---")
    print(format_gen_settings(gen_cfg))
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
            handle_playground_command(stripped, gen_cfg)
            continue

        text = generate(
            model,
            seed,
            tokenizer,
            gen_cfg["max_seq_len"],
            gen_cfg["generation_tokens"],
            gen_cfg["temperature"],
            gen_cfg["top_k"],
        )
        print(f"\n{text}\n")


def _param_names(model):
    return [name for name, _ in named_parameters(model)]


def save_optimizer_state_named(optimizer, model, path):
    if not isinstance(optimizer, Adam):
        raise TypeError("Only Adam optimizer state is supported.")
    names = _param_names(model)
    if len(names) != len(optimizer.params):
        raise ValueError("Model parameter count does not match optimizer.")
    state = {"adam_t": host_np.array([optimizer.t], dtype=host_np.int64)}
    for i, name in enumerate(names):
        m = optimizer.m[i]
        v = optimizer.v[i]
        if hasattr(m, "get"):
            m, v = m.get(), v.get()
        state[f"adam_m.{name}"] = host_np.asarray(m)
        state[f"adam_v.{name}"] = host_np.asarray(v)
    host_np.savez(path, **state)


def load_optimizer_state_named(optimizer, model, path):
    if not isinstance(optimizer, Adam):
        raise TypeError("Only Adam optimizer state is supported.")
    names = _param_names(model)
    with host_np.load(path) as data:
        optimizer.t = int(data["adam_t"][0])
        for i, name in enumerate(names):
            m_key = f"adam_m.{name}"
            v_key = f"adam_v.{name}"
            if m_key not in data.files:
                raise ValueError(f"Optimizer checkpoint missing {m_key}")
            optimizer.m[i] = np.asarray(data[m_key], dtype=optimizer.m[i].dtype)
            optimizer.v[i] = np.asarray(data[v_key], dtype=optimizer.v[i].dtype)


def save_training_checkpoint(model, optimizer, cfg, training_state, checkpoint_dir=None):
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    tmp_dir = checkpoint_dir.parent / "_checkpoint_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    save_weights(model, tmp_dir / "weights.npz")
    save_optimizer_state_named(optimizer, model, tmp_dir / "optimizer.npz")
    (tmp_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    training_state = dict(training_state)
    training_state["saved_at"] = time.time()
    (tmp_dir / "training.json").write_text(json.dumps(training_state, indent=2), encoding="utf-8")

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)
    return checkpoint_dir


def load_training_checkpoint(checkpoint_dir=None):
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "weights.npz"
    optim_path = checkpoint_dir / "optimizer.npz"
    training_path = checkpoint_dir / "training.json"

    for path in (config_path, weights_path, optim_path, training_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    training_state = json.loads(training_path.read_text(encoding="utf-8"))

    model = build_model(cfg)
    optimizer = Adam(model.parameters(), learning_rate=cfg["learning_rate"])
    load_weights(model, weights_path)
    load_optimizer_state_named(optimizer, model, optim_path)
    return model, optimizer, cfg, training_state


def checkpoint_exists(checkpoint_dir=None):
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    return (checkpoint_dir / "weights.npz").exists()
