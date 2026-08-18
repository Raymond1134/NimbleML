"""Direct preference optimization (DPO) on assistant-token mean NLL."""
from __future__ import annotations

from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def neg_log_sigmoid(logit: Tensor) -> Tensor:
    """Return ``-log sigmoid(x)`` = ``softplus(-x)`` with a stable forward."""
    x = np.asarray(logit.data, dtype=np.float64).reshape(-1)
    # log(1+exp(-x)): x>=0 → log1p(exp(-x)); x<0 → -x + log1p(exp(x))
    loss_np = np.where(x >= 0.0, np.log1p(np.exp(-x)), -x + np.log1p(np.exp(x)))
    out = Tensor(
        np.asarray(loss_np, dtype=np_backend.dtype).reshape(-1),
        () if loss_np.size == 1 else (int(loss_np.size),),
        requires_grad=logit.requires_grad,
        _children=(logit,),
        _op="neg_log_sigmoid",
    )
    saved = np.asarray(x, dtype=np.float64)

    def _backward():
        if out.grad is None or not logit.requires_grad:
            return
        g = np.asarray(out.grad, dtype=np.float64).reshape(-1)
        sig = 1.0 / (1.0 + np.exp(-saved))
        logit._accumulate_grad((g * (sig - 1.0)).astype(np_backend.dtype, copy=False).ravel())

    out._backward = _backward
    return out


def dpo_loss(nll_chosen, nll_rejected, ref_nll_chosen: float, ref_nll_rejected: float, *, beta: float = 0.1):
    """DPO from mean NLL of policy (tensors) and frozen ref (floats).

    ``logπ = -NLL``, so the DPO logit is
    ``β [ (NLL_l - NLL_w) + (NLL_ref_w - NLL_ref_l) ]``.
    """
    beta = float(beta)
    delta = (nll_rejected - nll_chosen) * beta
    const = beta * (float(ref_nll_chosen) - float(ref_nll_rejected))
    logit = delta + const
    return neg_log_sigmoid(logit)
