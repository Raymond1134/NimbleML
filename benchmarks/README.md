# NimbleML benchmarks

Pinned performance baselines for closing the gap to PyTorch. Re-run after every Phase 1–3 change in `todo.txt`.

## Reference config

Defined in `benchmarks/config.py` as `REFERENCE`:

| Field | Value |
|-------|-------|
| vocab | 4096 |
| d_model | 512 |
| heads | 8 |
| layers | 8 |
| ff_mult | 4 |
| batch | 4 |
| seq | 256 |
| warmup | 3 |
| runs | 5 |

`QUICK` is a smaller shape (256 vocab, 128d, 2 layers, seq 32) for fast smoke runs.

**Do not change `REFERENCE` casually** — update it only when you intentionally want a new baseline, then note the change in your commit message.

## Scripts

### Train throughput (`train_throughput.py`)

One full GPT training step: forward → cross-entropy → backward → grad clip → AdamW.

```bash
python benchmarks/train_throughput.py              # REFERENCE config, GPU if available
python benchmarks/train_throughput.py --quick      # smaller smoke config
python benchmarks/train_throughput.py --cpu        # force NumPy CPU backend
python benchmarks/train_throughput.py --no-torch   # NimbleML only
python benchmarks/train_throughput.py --json out.json
python benchmarks/train_throughput.py --profile          # autograd node count + budget
python benchmarks/train_throughput.py --fused-trunk      # single node for blocks + LN
python benchmarks/train_throughput.py --no-fused-blocks  # unfused reference path
```

Reports mean / p50 / p95 wall time, tokens/sec, and peak VRAM (GPU). Compares against a PyTorch reference model with pre-norm causal Transformer blocks, GELU FFN, tied embeddings, AdamW, and grad clipping.

**Target:** ~2× PyTorch tok/s on GPU after Phase 1–3 (fusion, fewer autograd nodes, AMP, CUDA graphs) before investing in FlashAttention.

### Forward only (`forward_only.py`)

Isolates where time goes on the forward path and measures autograd overhead:

| Benchmark | What it measures |
|-----------|------------------|
| `raw_attention` | CuPy/NumPy QKᵀ → softmax → @V (no autograd) |
| `mha_fwd_bwd` | Full `MultiHeadAttention` forward + backward |
| `gpt_embed` | Token + position embedding only |
| `gpt_blocks` | Embeddings + transformer blocks |
| `gpt_forward` | Full GPT forward |
| `gpt_forward_backward` | Full forward + fused tied CE backward |
| `autograd_nodes_forward` | Tensor node count on logits forward (lower is better) |
| `autograd_nodes_train` | Tensor node count on `compute_loss` train step |

```bash
python benchmarks/forward_only.py
python benchmarks/forward_only.py --quick --cpu
python benchmarks/forward_only.py --profile
python benchmarks/forward_only.py --fused-trunk
```

Benchmarks default to **fused transformer blocks** (`fused_blocks=True`). Use `--no-fused-blocks` for the unfused autograd path.

Use the MHA vs `raw_attention` ratio to estimate Python autograd overhead. Use `gpt_blocks` vs `gpt_forward` to see LM-head / norm cost.

## Environment

- `NIMBLEML_DEVICE=auto|cpu|gpu` — array backend (default `gpu` in benchmark scripts)
- `NIMBLEML_DTYPE=float32` — compute dtype (benchmarks pin float32)

### PyTorch comparison (optional)

The PyPI package is **`torch`**, not `pytorch`. NimbleML and PyTorch must run on the **same device** for the ratio to be meaningful.

**CPU-only** (fair comparison without CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python benchmarks/train_throughput.py --cpu
```

**GPU (CUDA)** — uninstall the CPU build first, then install the CUDA wheel:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want a `+cu124` (or similar) version and `True` for CUDA. Pick the CUDA version that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/).

If NimbleML is on GPU but PyTorch has no CUDA build, `train_throughput.py` skips the PyTorch row and prints a warning instead of a misleading ratio.

## Interpreting results

- **Train tok/s** is the primary metric; log it after each optimization pass.
- Each row shows `[gpu]` or `[cpu]` — verify both frameworks match before reading the ratio.
- **PyTorch / NimbleML ratio** above 1.0 means PyTorch is faster (expected early on).
- **Peak VRAM** helps catch accidental buffer copies (`_save_for_backward`).
- **Autograd node count** should drop sharply when fused transformer blocks land (Phase 2).
