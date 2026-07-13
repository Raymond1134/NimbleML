from .cross_entropy import CrossEntropyLoss
from .regression import L1Loss, MSELoss
from .sampled_cross_entropy import SampledCrossEntropyLoss

__all__ = ["CrossEntropyLoss", "SampledCrossEntropyLoss", "L1Loss", "MSELoss"]
