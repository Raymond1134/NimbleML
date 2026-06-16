# NimbleML

NimbleML is a lightweight Python machine learning library for building and training neural networks from scratch. It includes autograd tensors, standard layers, losses, optimizers, and optional GPU acceleration through CuPy.

## Installation

Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

Optional GPU support:

```bash
python -m pip install -e ".[gpu]"
```

Development setup:

```bash
python -m pip install -r requirements-dev.txt
```

## Quick Start

```python
from NimbleML import Dense, Tensor

x = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2), requires_grad=True)
layer = Dense(2, 3)
y = layer.forward(x)
```

Set execution device explicitly with `NIMBLEML_DEVICE=cpu` or `NIMBLEML_DEVICE=gpu`.

## Included Components

- Layers: `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dropout`, `Embedding`, `LayerNorm`
- Activations: `Relu`, `Softmax`
- Losses: `CrossEntropyLoss`
- Optimizers: `SGD`, `SGDM`, `NAG`, `RMSProp`, `Adam`
- Core: `Tensor`, `Module`, `Sequential`, `forward`, `parameters`, `train`, `eval`
- Schedulers: `StepLR`, `CosineAnnealing`, `LinearWarmup`

## Testing

```bash
python test.py
```

## Project Layout

- `NimbleML/`: library source package
- `test.py`: unit and integration tests
- `pyproject.toml`: package metadata and build configuration
- `requirements.txt`: runtime dependencies
- `requirements-dev.txt`: development tooling dependencies
