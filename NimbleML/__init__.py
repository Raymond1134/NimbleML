"""Package exports and public API surface."""

from .utils.tensor import Tensor
from .utils.np_backend import device, np, set_device, using_gpu
from .utils.saveload import load, save
from .core import eval, forward, parameters, train
from .layers import Conv2D, Dense, Dropout, Embedding, Flatten, MaxPool2D
from .neural_network import Module, Sequential
from .activations import Relu, Softmax
from .losses import CrossEntropyLoss
from .optimizers import Adam, NAG, Optimizer, RMSProp, SGD, SGDM

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
    "Optimizer",
    "SGD",
    "SGDM",
    "NAG",
    "RMSProp",
    "Adam",
    "save",
    "load",
]
