"""Direct preference optimization (DPO).

Use per-pair **summed** sequence NLL (see :func:`tied_sequence_nll`); a
batch-mean NLL collapses every pair in a micro-batch into one logit.
"""
from __future__ import annotations

import numpy as host_np

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
    """DPO from per-sequence summed NLL of the policy (tensors) and frozen ref (floats).

    ``logπ = -NLL``, so the DPO logit is
    ``β [ (NLL_l - NLL_w) + (NLL_ref_w - NLL_ref_l) ]``.
    """
    beta = float(beta)
    delta = (nll_rejected - nll_chosen) * beta
    const = beta * (float(ref_nll_chosen) - float(ref_nll_rejected))
    logit = delta + const
    return neg_log_sigmoid(logit)


def dpo_pair_terms(nll_chosen, nll_rejected, ref_chosen, ref_rejected, *, beta: float = 0.1) -> dict:
    """Host-side DPO for arrays of summed NLL (one entry per pair, float64).

    Returns per-pair ``loss``, the gradients ``d_chosen`` / ``d_rejected`` of the
    loss w.r.t. each summed NLL (feed them to :func:`weighted_sum`), and the
    implicit rewards ``β (logπ - logπ_ref)``.
    """
    c = host_np.asarray(nll_chosen, dtype=host_np.float64)
    r = host_np.asarray(nll_rejected, dtype=host_np.float64)
    rc = host_np.asarray(ref_chosen, dtype=host_np.float64)
    rr = host_np.asarray(ref_rejected, dtype=host_np.float64)
    beta = float(beta)
    z = beta * ((r - c) + (rc - rr))
    loss = host_np.logaddexp(0.0, -z)
    one_minus_sig = 1.0 / (1.0 + host_np.exp(z))
    return {
        "loss": loss,
        "d_chosen": beta * one_minus_sig,
        "d_rejected": -beta * one_minus_sig,
        "reward_chosen": beta * (rc - c),
        "reward_rejected": beta * (rr - r),
    }
