# NimbleML

A lightweight Python machine learning library for building and training neural networks from scratch. NimbleML includes autograd tensors, common layers, losses, optimizers, and optional GPU support via CuPy.

## Install

CPU only:

```bash
python -m pip install numpy
```

GPU (NVIDIA CUDA, optional — auto-detected when available):

```bash
python -m pip install numpy cupy-cuda11x
```

Or install from the project root:

```bash
python -m pip install -r requirements.txt
```

Force CPU or GPU with the environment variable `NIMBLEML_DEVICE=cpu` or `NIMBLEML_DEVICE=gpu`.

## What's included

**Layers:** `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dropout`

**Activations:** `Relu`, `Softmax`

**Losses:** `CrossEntropyLoss`

**Optimizers:** `SGD`, `SGDM`, `NAG`, `RMSProp`, `Adam`

**Core:** `Tensor` (autograd), `Module`, `Sequential`, `forward`, `parameters`, `train` / `eval`, NumPy/CuPy backend (`device`, `using_gpu`, `set_device`)

## Tests

```bash
python test.py
```

## Project layout

- `NimbleML/` — library source
- `test.py` — unit tests
