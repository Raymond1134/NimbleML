"""Package exports and public API surface."""

from .utils.tensor import Tensor
from .utils.np_backend import device, np, set_device, using_gpu
from .utils.saveload import load, save
from .utils.clip_grad import clip_grad_norm_
from .core import eval, forward, parameters, train
from .layers import Conv2D, Dense, Dropout, Embedding, Flatten, MaxPool2D
from .neural_network import Module, Sequential
from .activations import Relu, Softmax
from .losses import CrossEntropyLoss, L1Loss, MSELoss
from .optimizers import Adam, AdamW, NAG, Optimizer, RMSProp, SGD, SGDM
from .models import GPT
from .metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_f1, r2_score

__all__ = [
    "Tensor",
    "np",
    "device",
    "using_gpu",
    "set_device",
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
]
