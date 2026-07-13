# NimbleML

A lightweight Python machine learning library for building and training neural networks from scratch.

NimbleML provides autograd `Tensor`s, a `Module` API, layers, losses, optimizers, and optional GPU acceleration via CuPy. It is **alpha** software focused on **GPT-style language modeling**: causal transformers, tied embeddings, fused training kernels, and throughput benchmarks against PyTorch.

**Requirements:** Python 3.9+, NumPy, a C++ toolchain (the `nimbleml_native` extension is **required** — no Python fallback). GPU support is optional (CuPy).

## Installation

Windows: open a **x64 Native Tools** / `vcvars64` shell (Visual Studio Build Tools), then:

```bash
python -m pip install -e ".[dev]"
python -c "import nimbleml_native; print('native ok')"
```

Optional extras:

```bash
python -m pip install -e ".[gpu]"        # CuPy GPU backend
python -m pip install -e ".[dev]"        # pytest + ruff (+ builds native)
python -m pip install -e ".[gpu,dev]"    # both
```

CUDA device FlashAttention + kernels: set `NIMBLEML_WITH_CUDA=ON` in `[tool.scikit-build.cmake.define]` (needs `nvcc`). Include Ada (`89`) in `CUDAARCHS` for L40S.

Chatbot recipe: [examples/chatbot/README.md](examples/chatbot/README.md).  
Performance stack: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Quick start

### Autograd and a dense layer

```python
from NimbleML import Dense, Tensor

x = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2), requires_grad=True)
layer = Dense(2, 3)
y = layer.forward(x)
y.sum().backward()
```

### Train a small GPT (manual loop)

```python
from NimbleML import GPT, AdamW, Tensor, clip_grad_norm_
from NimbleML.utils.np_backend import np

model = GPT(
    vocab_size=256,
    d_model=64,
    num_heads=4,
    num_layers=2,
    max_seq_len=32,
    fused_blocks=True,
)
optimizer = AdamW(model.parameters(), learning_rate=3e-4)

input_ids = Tensor.from_int64(np.arange(16, dtype=np.int64).reshape(2, 8), (2, 8))
labels = Tensor.from_int64(np.random.randint(0, 256, size=16, dtype=np.int64), (2, 8))

optimizer.zero_grad()
loss = model.compute_loss(input_ids, labels)  # fused tied cross-entropy
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Prefer `compute_loss()` over `CrossEntropyLoss(model(input_ids), labels)` during training — it fuses the LM head with cross-entropy and avoids materializing a full vocab softmax.

### Train with a DataLoader

```python
from NimbleML.data import DataLoader, TokenLMDataset, load_text

ids, _, _ = load_text("NimbleML/data/text/tiny_corpus.txt")
loader = DataLoader(TokenLMDataset(ids, seq_len=32), batch_size=8, shuffle=True)

for input_ids, labels in loader:
    optimizer.zero_grad()
    loss = model.compute_loss(input_ids, labels)
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Resume training

```python
from NimbleML import save_checkpoint, load_checkpoint

save_checkpoint("ckpt.npz", model, optimizer, scheduler, step=1000)
load_checkpoint("ckpt.npz", model, optimizer, scheduler)
```

`save` / `load` handle weights only. `save_checkpoint` / `load_checkpoint` also persist optimizer and scheduler state.

## Environment

| Variable | Values | Default |
|----------|--------|---------|
| `NIMBLEML_DEVICE` | `auto`, `cpu`, `gpu` | `auto` |
| `NIMBLEML_DTYPE` | `float16`, `float32`, `float64` | `float32` |
| `NIMBLEML_SDPA` | `auto`, `matmul`, `flash` | `auto` |
| `NIMBLEML_WITH_CUDA` | `ON` / off | off (build flag) |
| `CUPY_TF32` | `1` / `0` | unset |

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the full list.

## API overview

### `from NimbleML import ...`

| Area | Symbols |
|------|---------|
| Core | `Tensor`, `Module`, `Sequential`, `forward`, `parameters`, `train`, `eval` |
| I/O | `save`, `load`, `save_checkpoint`, `load_checkpoint` |
| Backend | `np`, `device`, `set_device`, `using_gpu` |
| Layers | `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dropout`, `Embedding` |
| Activations | `Relu`, `Softmax` |
| Losses | `CrossEntropyLoss`, `MSELoss`, `L1Loss` |
| Optimizers | `SGD`, `SGDM`, `NAG`, `RMSProp`, `Adam`, `AdamW` |
| Models | `GPT` |
| Metrics | `accuracy_score`, `precision_recall_f1`, `mean_squared_error`, `mean_absolute_error`, `r2_score` |
| Training | `clip_grad_norm_` |

### Submodule imports

| Area | Import from | Examples |
|------|-------------|----------|
| Layers & activations | `NimbleML.layers`, `NimbleML.activations` | `LayerNorm`, `RMSNorm`, `Gelu` |
| Transformer | `NimbleML.neural_network` | `TransformerBlock`, `FusedTransformerBlock`, `MultiHeadAttention` |
| Losses | `NimbleML.losses` | `SampledCrossEntropyLoss` |
| Schedulers | `NimbleML.optimizers` | `StepLR`, `LinearWarmup`, `CosineAnnealingLR` |
| Data | `NimbleML.data` | `Dataset`, `DataLoader`, `TokenLMDataset`, `BPETokenizer` |
| Utils | `NimbleML.utils` | `gradcheck`, `autograd_profile` |

## Testing

```bash
python test.py
```

With pytest (`pip install -e ".[dev]"`), run tests locally from a `tests/` checkout (that directory is not in the public repo).

## Project layout

```
NimbleML/
  activations/       Relu, Softmax, Gelu
  data/              Dataset, DataLoader, text tokenizers
  kernels/           Fused GPU-friendly ops (CE, GELU, RMSNorm, …)
  layers/            Dense, Conv2D, norm, embedding, …
  losses/            Cross-entropy, regression, sampled CE
  models/            GPT
  neural_network/    Attention, transformer blocks, Module API
  optimizers/        SGD family, Adam/AdamW, schedulers
  utils/             Tensor autograd, backend, checkpoints, profiling
docs/                Performance notes
```

Install dependencies from `pyproject.toml` (runtime + optional extras):

```bash
pip install -e .              # numpy, pybind11
pip install -e ".[dev,gpu]"   # pytest, ruff, cupy (CUDA 12x wheel)
```

## License

MIT — see [LICENSE](LICENSE).
