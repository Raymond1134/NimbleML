from .utils.tensor import Tensor
from .utils.np_backend import device, np, set_device, using_gpu
from .utils.grad_mode import enable_grad, is_grad_enabled, no_grad
from .utils.saveload import load, load_checkpoint, save, save_checkpoint
from .utils.clip_grad import clip_grad_norm_
from .core import eval, forward, parameters, train
from .layers import Conv2D, Dense, Dropout, Embedding, Flatten, MaxPool2D
from .neural_network import Module, ModuleList, Sequential
from .activations import Relu, Softmax
from .losses import CrossEntropyLoss, L1Loss, MSELoss
from .optimizers import Adam, AdamW, NAG, Optimizer, RMSProp, SGD, SGDM
from .trainer import Trainer
from .models import GPT
from .metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_f1, r2_score

# Required native extension — fail fast if not built.
from ._native_loader import native as _native  # noqa: F401

# GPU allocator / TF32 after the package graph is loaded (see np_backend).
if using_gpu:
    from .utils.np_backend import configure_gpu_runtime as _configure_gpu_runtime

    _configure_gpu_runtime(verbose=False)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Tensor",
    "np",
    "device",
    "using_gpu",
    "set_device",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "forward",
    "parameters",
    "train",
    "eval",
    "Conv2D",
    "Dense",
    "Dropout",
    "Embedding",
    "Flatten",
    "MaxPool2D",
    "Module",
    "ModuleList",
    "Sequential",
    "Relu",
    "Softmax",
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
    "Optimizer",
    "SGD",
    "SGDM",
    "NAG",
    "RMSProp",
    "Adam",
    "AdamW",
    "GPT",
    "accuracy_score",
    "precision_recall_f1",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "clip_grad_norm_",
    "save",
    "load",
    "save_checkpoint",
    "load_checkpoint",
    "Trainer",
]
