"""Numerical parity: NimbleML kernels vs naive NumPy references (CPU and GPU)."""
from __future__ import annotations

import importlib
import os

import numpy as host_np
import pytest

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_KERNEL_MODULES = (
    "NimbleML.kernels.embedding_scatter",
    "NimbleML.kernels.fused_crossentropy",
    "NimbleML.kernels.fused_gelu",
    "NimbleML.kernels.fused_rmsnorm",
    "NimbleML.kernels.fused_tied_crossentropy",
    "NimbleML.kernels.sampled_softmax",
)


def _activate_backend(device: str):
    """Reload np_backend and kernel modules so ``np`` matches *device*."""
    os.environ["NIMBLEML_DEVICE"] = device
    import NimbleML.utils.np_backend as nb

    importlib.reload(nb)
    for name in _KERNEL_MODULES:
        importlib.reload(importlib.import_module(name))
    return nb.np, nb.using_gpu


def _to_host(arr) -> host_np.ndarray:
    get = getattr(arr, "get", None)
    return host_np.asarray(get() if get is not None else arr)


def _assert_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    host_np.testing.assert_allclose(
        _to_host(actual),
        host_np.asarray(expected),
        rtol=rtol,
        atol=atol,
    )


@pytest.fixture
def cpu_np():
    np, _ = _activate_backend("cpu")
    yield np
    _activate_backend("cpu")


@pytest.fixture
def gpu_np():
    if not _gpu_available():
        pytest.skip("CUDA GPU not available")
    np, using_gpu = _activate_backend("gpu")
    if not using_gpu:
        pytest.skip("CuPy GPU backend could not be initialized")
    yield np
    _activate_backend("cpu")


# ---------------------------------------------------------------------------
# Naive references (host NumPy only)
# ---------------------------------------------------------------------------


def _ref_gelu_forward(x):
    k = host_np.sqrt(2.0 / host_np.pi)
    tanh_u = host_np.tanh(k * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1.0 + tanh_u), tanh_u


def _ref_gelu_backward(grad_out, x, tanh_u):
    k = host_np.sqrt(2.0 / host_np.pi)
    du_dx = k * (1.0 + 0.134145 * x * x)
    sech2 = 1.0 - tanh_u * tanh_u
    grad_x = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du_dx
    return grad_out * grad_x


def _ref_rmsnorm_forward(x, gamma, epsilon=1e-5):
    ms = host_np.mean(x * x, axis=-1, keepdims=True)
    rms = host_np.sqrt(ms + epsilon)
    out = (x / rms) * gamma
    return out, x, ms, rms


def _ref_rmsnorm_backward(grad_out, x, gamma, ms, rms, epsilon=1e-5):
    d = x.shape[-1]
    x_hat = x / rms
    grad_gamma = host_np.sum(grad_out * x_hat, axis=0)
    grad_x_hat = grad_out * gamma
    grad_ms = host_np.sum(
        grad_x_hat * x * (-0.5) * (ms + epsilon) ** (-1.5),
        axis=-1,
        keepdims=True,
    )
    grad_x_from_ms = (2.0 / d) * x * grad_ms
    grad_x_direct = grad_x_hat / rms
    return grad_x_direct + grad_x_from_ms, grad_gamma


def _ref_crossentropy_forward(logits, labels):
    batch_size = logits.shape[0]
    max_vals = host_np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_vals
    sum_exp = host_np.sum(host_np.exp(shifted), axis=1, keepdims=True)
    log_sum_exp = max_vals.ravel() + host_np.log(sum_exp.ravel())
    row_idx = host_np.arange(batch_size, dtype=host_np.int64)
    correct = logits[row_idx, labels]
    per_sample = log_sum_exp - correct
    loss = float(host_np.sum(per_sample) / batch_size)
    return loss, max_vals, sum_exp


def _ref_crossentropy_backward(grad_scale, logits, labels, max_vals, sum_exp):
    batch_size = logits.shape[0]
    shifted = logits - max_vals
    probs = host_np.exp(shifted) / sum_exp
    grad = probs.copy()
    row_idx = host_np.arange(batch_size, dtype=host_np.int64)
    grad[row_idx, labels] -= 1.0
    grad /= batch_size
    return grad * float(grad_scale)


def _ref_embedding_lookup(weights, ids):
    flat = host_np.asarray(ids, dtype=host_np.int64).reshape(-1)
    out = weights[flat]
    id_shape = host_np.asarray(ids).shape
    if id_shape:
        return out.reshape(*id_shape, weights.shape[1])
    return out.reshape(-1, weights.shape[1])


def _ref_embedding_scatter_add(grad_w, ids, grad_out):
    out = host_np.array(grad_w, copy=True)
    ids_h = host_np.asarray(ids, dtype=host_np.int64).reshape(-1)
    grad_h = host_np.asarray(grad_out, dtype=host_np.float32).reshape(ids_h.size, -1)
    host_np.add.at(out, ids_h, grad_h)
    return out


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------


class TestFusedGelu:
    def test_forward_backward_cpu(self, cpu_np):
        from NimbleML.kernels.fused_gelu import fused_gelu_backward, fused_gelu_forward

        rng = host_np.random.default_rng(0)
        x = rng.standard_normal((4, 16)).astype(host_np.float32)
        out_ref, tanh_ref = _ref_gelu_forward(x)
        out_ker, tanh_ker = fused_gelu_forward(x)
        _assert_close(out_ker, out_ref)
        _assert_close(tanh_ker, tanh_ref)

        grad_out = rng.standard_normal(x.shape).astype(host_np.float32)
        _assert_close(
            fused_gelu_backward(grad_out, x, tanh_ker),
            _ref_gelu_backward(grad_out, x, tanh_ref),
        )

    @pytest.mark.gpu
    def test_forward_backward_gpu(self, gpu_np):
        from NimbleML.kernels.fused_gelu import fused_gelu_backward, fused_gelu_forward

        rng = host_np.random.default_rng(1)
        x = rng.standard_normal((4, 16)).astype(host_np.float32)
        out_ref, tanh_ref = _ref_gelu_forward(x)
        out_ker, tanh_ker = fused_gelu_forward(x)
        _assert_close(out_ker, out_ref, rtol=1e-4, atol=1e-4)
        _assert_close(tanh_ker, tanh_ref, rtol=1e-4, atol=1e-4)

        grad_out = rng.standard_normal(x.shape).astype(host_np.float32)
        _assert_close(
            fused_gelu_backward(grad_out, x, tanh_ker),
            _ref_gelu_backward(grad_out, x, tanh_ref),
            rtol=1e-4,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestFusedRmsNorm:
    def test_forward_backward_cpu(self, cpu_np):
        from NimbleML.kernels.fused_rmsnorm import fused_rmsnorm_backward, fused_rmsnorm_forward

        rng = host_np.random.default_rng(2)
        x = rng.standard_normal((3, 8, 32)).astype(host_np.float32)
        gamma = rng.standard_normal((32,)).astype(host_np.float32)
        out_ref, x_ref, ms_ref, rms_ref = _ref_rmsnorm_forward(x, gamma)
        out_ker, x_ker, ms_ker, rms_ker = fused_rmsnorm_forward(x, gamma)
        _assert_close(out_ker, out_ref)
        _assert_close(ms_ker, ms_ref)
        _assert_close(rms_ker, rms_ref)

        grad_out = rng.standard_normal(out_ref.shape).astype(host_np.float32)
        gx_ref, gg_ref = _ref_rmsnorm_backward(grad_out, x_ref, gamma, ms_ref, rms_ref)
        gx_ker, gg_ker = fused_rmsnorm_backward(grad_out, x_ker, gamma, ms_ker, rms_ker)
        _assert_close(gx_ker, gx_ref)
        _assert_close(gg_ker, gg_ref)

    @pytest.mark.gpu
    def test_forward_backward_gpu(self, gpu_np):
        from NimbleML.kernels.fused_rmsnorm import fused_rmsnorm_backward, fused_rmsnorm_forward

        rng = host_np.random.default_rng(3)
        x = rng.standard_normal((3, 8, 32)).astype(host_np.float32)
        gamma = rng.standard_normal((32,)).astype(host_np.float32)
        out_ref, x_ref, ms_ref, rms_ref = _ref_rmsnorm_forward(x, gamma)
        out_ker, x_ker, ms_ker, rms_ker = fused_rmsnorm_forward(x, gamma)
        _assert_close(out_ker, out_ref, rtol=1e-4, atol=1e-4)

        grad_out = rng.standard_normal(out_ref.shape).astype(host_np.float32)
        gx_ref, gg_ref = _ref_rmsnorm_backward(grad_out, x_ref, gamma, ms_ref, rms_ref)
        gx_ker, gg_ker = fused_rmsnorm_backward(grad_out, x_ker, gamma, ms_ker, rms_ker)
        _assert_close(gx_ker, gx_ref, rtol=1e-4, atol=1e-4)
        _assert_close(gg_ker, gg_ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------


class TestFusedCrossEntropy:
    def test_forward_backward_cpu(self, cpu_np):
        from NimbleML.kernels.fused_crossentropy import (
            fused_crossentropy_backward,
            fused_crossentropy_forward,
        )

        rng = host_np.random.default_rng(4)
        logits = rng.standard_normal((5, 11)).astype(host_np.float32)
        labels = rng.integers(0, 11, size=(5,), dtype=host_np.int64)
        loss_ref, max_ref, sum_ref = _ref_crossentropy_forward(logits, labels)
        loss_ker, logits_ker, max_ker, sum_ker = fused_crossentropy_forward(logits, labels)
        assert host_np.isclose(loss_ker, loss_ref, rtol=1e-5, atol=1e-5)
        _assert_close(max_ker, max_ref)
        _assert_close(sum_ker, sum_ref)

        grad_ref = _ref_crossentropy_backward(1.0, logits, labels, max_ref, sum_ref)
        grad_ker = fused_crossentropy_backward(1.0, logits_ker, labels, max_ker, sum_ker)
        _assert_close(grad_ker, grad_ref)

    @pytest.mark.gpu
    def test_forward_backward_gpu(self, gpu_np):
        from NimbleML.kernels.fused_crossentropy import (
            fused_crossentropy_backward,
            fused_crossentropy_forward,
        )

        rng = host_np.random.default_rng(5)
        logits = rng.standard_normal((5, 11)).astype(host_np.float32)
        labels = rng.integers(0, 11, size=(5,), dtype=host_np.int64)
        loss_ref, max_ref, sum_ref = _ref_crossentropy_forward(logits, labels)
        loss_ker, logits_ker, max_ker, sum_ker = fused_crossentropy_forward(logits, labels)
        assert host_np.isclose(loss_ker, loss_ref, rtol=1e-4, atol=1e-4)

        grad_ref = _ref_crossentropy_backward(1.0, logits, labels, max_ref, sum_ref)
        grad_ker = fused_crossentropy_backward(1.0, logits_ker, labels, max_ker, sum_ker)
        _assert_close(grad_ker, grad_ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Embedding gather / scatter
# ---------------------------------------------------------------------------


class TestEmbeddingScatter:
    def test_lookup_scatter_cpu(self, cpu_np):
        from NimbleML.kernels.embedding_scatter import embedding_lookup, embedding_scatter_add

        rng = host_np.random.default_rng(6)
        weights = rng.standard_normal((20, 8)).astype(host_np.float32)
        ids = rng.integers(0, 20, size=(2, 5), dtype=host_np.int64)
        out_ref = _ref_embedding_lookup(weights, ids)
        out_ker = embedding_lookup(weights, ids)
        _assert_close(out_ker, out_ref)

        grad_out = rng.standard_normal(out_ref.shape).astype(host_np.float32)
        grad_w = host_np.zeros_like(weights)
        grad_ref = _ref_embedding_scatter_add(grad_w, ids.reshape(-1), grad_out.reshape(-1, 8))
        grad_ker = embedding_scatter_add(host_np.zeros_like(weights), ids.reshape(-1), grad_out)
        _assert_close(grad_ker, grad_ref)

    @pytest.mark.gpu
    def test_lookup_scatter_gpu(self, gpu_np):
        from NimbleML.kernels.embedding_scatter import embedding_lookup, embedding_scatter_add

        rng = host_np.random.default_rng(7)
        weights = rng.standard_normal((20, 8)).astype(host_np.float32)
        ids = rng.integers(0, 20, size=(2, 5), dtype=host_np.int64)
        out_ref = _ref_embedding_lookup(weights, ids)
        out_ker = embedding_lookup(weights, ids)
        _assert_close(out_ker, out_ref, rtol=1e-4, atol=1e-4)

        grad_out = rng.standard_normal(out_ref.shape).astype(host_np.float32)
        grad_ker = embedding_scatter_add(
            host_np.zeros_like(weights),
            ids.reshape(-1),
            grad_out,
        )
        grad_ref = _ref_embedding_scatter_add(
            host_np.zeros_like(weights),
            ids.reshape(-1),
            grad_out.reshape(-1, 8),
        )
        _assert_close(grad_ker, grad_ref, rtol=1e-4, atol=1e-4)


class TestFusedTiedCrossEntropy:
    def test_matches_untied_path_cpu(self, cpu_np):
        from NimbleML.kernels.fused_tied_crossentropy import (
            fused_tied_crossentropy_backward,
            fused_tied_crossentropy_forward,
        )

        rng = host_np.random.default_rng(8)
        hidden = rng.standard_normal((6, 16)).astype(host_np.float32)
        weights = rng.standard_normal((11, 16)).astype(host_np.float32)
        labels = rng.integers(0, 11, size=(6,), dtype=host_np.int64)

        logits = hidden @ weights.T
        loss_ref, max_ref, sum_ref = _ref_crossentropy_forward(logits, labels)
        loss_ker, h_ker, w_ker, max_ker, sum_ker = fused_tied_crossentropy_forward(hidden, weights, labels)
        assert host_np.isclose(loss_ker, loss_ref, rtol=1e-5, atol=1e-5)

        gh_ref_logits = _ref_crossentropy_backward(1.0, logits, labels, max_ref, sum_ref)
        gh_ker, gw_ker = fused_tied_crossentropy_backward(1.0, h_ker, w_ker, labels, max_ker, sum_ker)
        _assert_close(gh_ker, gh_ref_logits @ weights)
        _assert_close(gw_ker, gh_ref_logits.T @ hidden)

    @pytest.mark.gpu
    def test_matches_untied_path_gpu(self, gpu_np):
        from NimbleML.kernels.fused_tied_crossentropy import (
            fused_tied_crossentropy_backward,
            fused_tied_crossentropy_forward,
        )

        rng = host_np.random.default_rng(9)
        hidden = rng.standard_normal((6, 16)).astype(host_np.float32)
        weights = rng.standard_normal((11, 16)).astype(host_np.float32)
        labels = rng.integers(0, 11, size=(6,), dtype=host_np.int64)

        logits = hidden @ weights.T
        loss_ref, max_ref, sum_ref = _ref_crossentropy_forward(logits, labels)
        loss_ker, h_ker, w_ker, max_ker, sum_ker = fused_tied_crossentropy_forward(hidden, weights, labels)
        assert host_np.isclose(loss_ker, loss_ref, rtol=1e-4, atol=1e-4)

        gh_ref_logits = _ref_crossentropy_backward(1.0, logits, labels, max_ref, sum_ref)
        gh_ker, gw_ker = fused_tied_crossentropy_backward(1.0, h_ker, w_ker, labels, max_ker, sum_ker)
        _assert_close(gh_ker, gh_ref_logits @ weights, rtol=1e-4, atol=1e-4)
        _assert_close(gw_ker, gh_ref_logits.T @ hidden, rtol=1e-4, atol=1e-4)


class TestSampledSoftmax:
    def test_matches_full_ce_with_all_negatives_cpu(self, cpu_np):
        from NimbleML.kernels.fused_crossentropy import (
            fused_crossentropy_backward,
            fused_crossentropy_forward,
        )
        from NimbleML.kernels.sampled_softmax import (
            sampled_softmax_backward,
            sampled_softmax_forward,
        )

        rng = host_np.random.default_rng(10)
        vocab = 11
        batch = 5
        logits = rng.standard_normal((batch, vocab)).astype(host_np.float32)
        labels = rng.integers(0, vocab, size=(batch,), dtype=host_np.int64)
        negatives = host_np.stack(
            [host_np.array([j for j in range(vocab) if j != int(labels[i])], dtype=host_np.int64) for i in range(batch)]
        )

        loss_ref, _, max_ref, sum_ref = fused_crossentropy_forward(logits, labels)
        loss_ker, _, cand_ker, max_ker, sum_ker = sampled_softmax_forward(logits, labels, negatives)
        assert host_np.isclose(loss_ker, loss_ref, rtol=1e-5, atol=1e-5)

        grad_ref = fused_crossentropy_backward(1.0, logits, labels, max_ref, sum_ref)
        grad_ker = sampled_softmax_backward(1.0, logits, cand_ker, max_ker, sum_ker)
        _assert_close(grad_ker, grad_ref)

    @pytest.mark.gpu
    def test_sampled_forward_backward_gpu(self, gpu_np):
        from NimbleML.kernels.sampled_softmax import (
            sample_negative_indices,
            sampled_softmax_backward,
            sampled_softmax_forward,
        )

        rng = host_np.random.default_rng(11)
        vocab = 64
        batch = 6
        num_samples = 8
        logits = rng.standard_normal((batch, vocab)).astype(host_np.float32)
        labels = rng.integers(0, vocab, size=(batch,), dtype=host_np.int64)
        negatives = sample_negative_indices(vocab, labels, num_samples, rng=rng)

        loss_ker, logits_ker, cand_ker, max_ker, sum_ker = sampled_softmax_forward(logits, labels, negatives)
        assert host_np.isfinite(loss_ker)
        grad_ker = sampled_softmax_backward(1.0, logits_ker, cand_ker, max_ker, sum_ker)
        assert grad_ker.shape == (batch, vocab)
        assert host_np.any(grad_ker != 0)
