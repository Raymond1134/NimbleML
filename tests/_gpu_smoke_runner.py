"""Isolated GPU training smoke run (fresh interpreter, no pytest import side effects)."""

from __future__ import annotations

import gc
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["NIMBLEML_DEVICE"] = "gpu"
os.environ["NIMBLEML_DTYPE"] = "float32"

import numpy as host_np  # noqa: E402

from NimbleML.losses import CrossEntropyLoss  # noqa: E402
from NimbleML.models import GPT  # noqa: E402
from NimbleML.optimizers import AdamW  # noqa: E402
from NimbleML.utils.clip_grad import clip_grad_norm_  # noqa: E402
from NimbleML.utils.np_backend import apply_runtime_config, using_gpu  # noqa: E402
from toyGPT.data import random_batch  # noqa: E402
from toyGPT.fineweb import load_token_bin  # noqa: E402
from toyGPT.train_utils import adamw_param_groups  # noqa: E402

apply_runtime_config("gpu", "float32")

if not using_gpu:
    raise SystemExit("CUDA GPU not available")

cache = ROOT / "toyGPT" / "data" / "cache" / "encoded" / "fineweb-edu"
train_bin = next(cache.glob("*/train.bin"), None)
if train_bin is None:
    raise SystemExit("No encoded FineWeb corpus cache found")

train_ids = load_token_bin(train_bin)
model = GPT(16384, 512, 8, 12, 256, ff_mult=4)
opt = AdamW(
    adamw_param_groups(model, lr=3e-4, weight_decay=0.1),
    learning_rate=3e-4,
    weight_decay=0.1,
)
loss_fn = CrossEntropyLoss()
rng = host_np.random.default_rng(0)
batch_size = 8

import cupy  # noqa: E402

for step in range(1, 21):
    opt.zero_grad(set_to_none=True)
    inp, tgt = random_batch(train_ids, batch_size=batch_size, seq_len=256, rng=rng)
    logits = model.forward(inp)
    loss = loss_fn(logits, tgt)
    loss.backward()
    lv = float(loss.data[0])
    if not math.isfinite(lv):
        raise SystemExit(f"non-finite loss at step {step}: {lv}")
    del logits, loss, inp, tgt
    clip_grad_norm_(model.parameters(), 1.0)
    model.clear_pos_encoding_cache()
    opt.step()
    gc.collect()
    cupy.cuda.Device().synchronize()

print("OK: 20 steps 12L model batch_size=8")
