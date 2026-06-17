"""Mandatory correctness tests for NimbleML core functionality."""

import os

os.environ["NIMBLEML_DEVICE"] = "cpu"

from pathlib import Path

from NimbleML.activations import Softmax
from NimbleML.layers import Embedding, MaxPool2D
from NimbleML.layers.conv2d import Conv2D
from NimbleML.layers.dense import Dense
from NimbleML.models.gpt import GPT
from NimbleML.neural_network.attention import Attention, make_causal_mask
from NimbleML.optimizers import SGD, StepLR
from NimbleML.utils.clip_grad import clip_grad_norm_
from NimbleML.utils.np_backend import np, set_dtype
from NimbleML.utils.saveload import load, named_parameters, save
from NimbleML.utils.tensor import Tensor

# Default training path: GPU + float32 (see toyGPT/gpt_toy_config.toml).
set_dtype("float32")


def _global_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = np.asarray(p.grad)
        total += float(np.sum(g * g))
    return total ** 0.5


def test_tensor_broadcast_backward():
    a = Tensor([1, 2], (2, 1), requires_grad=True)
    b = Tensor([10, 20, 30], (1, 3), requires_grad=True)
    (a + b).sum().backward()
    assert np.allclose(np.asarray(a.grad), np.array([3, 3], dtype=np.float64))
    assert np.allclose(np.asarray(b.grad), np.array([2, 2, 2], dtype=np.float64))


def test_dense_forward_backward():
    layer = Dense(4, 3)
    x = Tensor(np.linspace(0.1, 0.8, 8, dtype=np.float64), (2, 4), requires_grad=True)
    out = layer.forward(x)
    assert out.shape == (2, 3)
    out.sum().backward()
    assert x.grad is not None
    assert layer.weights.grad is not None
    assert layer.biases.grad is not None


def test_conv2d_forward_backward():
    layer = Conv2D(1, 2, kernel_size=3, stride=1, padding=0, bias=True)
    x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    out = layer.forward(x)
    assert out.shape == (1, 2, 2, 2)
    out.sum().backward()
    assert x.grad is not None
    assert layer.weights.grad is not None
    assert layer.biases.grad is not None


def test_maxpool2d_backward_mask():
    layer = MaxPool2D(kernel_size=2, stride=2)
    x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    out = layer.forward(x)
    out.sum().backward()
    expected = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.float64)
    assert np.allclose(np.asarray(x.grad), expected)


def test_embedding_backward_accumulates():
    layer = Embedding(vocab_size=10, embed_dim=4)
    ids = [[0, 2, 2], [1, 3, 0]]
    out = layer.forward(ids)
    out.sum().backward()
    assert layer.weights.grad is not None
    grad = np.asarray(layer.weights.grad).reshape(10, 4)
    # token 2 appears twice, token 1 once, token 9 never.
    assert np.allclose(grad[2], np.full(4, 2.0))
    assert np.allclose(grad[1], np.full(4, 1.0))
    assert np.allclose(grad[9], np.zeros(4))


def test_softmax_normalization_and_grad():
    logits = Tensor(np.linspace(-1, 1, 18, dtype=np.float64), (2, 3, 3), requires_grad=True)
    probs = Softmax()(logits)
    row_sums = np.asarray(probs.data).reshape(2, 3, 3).sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    probs.sum().backward()
    assert logits.grad is not None


def test_attention_shape_with_causal_mask():
    batch, seq_len, d_k = 2, 4, 8
    rng = np.random.default_rng(0)
    q = Tensor(rng.standard_normal((batch, seq_len, d_k)).ravel(), (batch, seq_len, d_k), requires_grad=True)
    k = Tensor(rng.standard_normal((batch, seq_len, d_k)).ravel(), (batch, seq_len, d_k), requires_grad=True)
    v = Tensor(rng.standard_normal((batch, seq_len, d_k)).ravel(), (batch, seq_len, d_k), requires_grad=True)
    out = Attention(d_k).forward(q, k, v, mask=make_causal_mask(seq_len))
    assert out.shape == (batch, seq_len, d_k)
    out.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_multi_head_attention_forward_backward():
    from NimbleML.neural_network.attention import MultiHeadAttention, causal_mask_tensor

    batch, seq_len, d_model, num_heads = 2, 8, 32, 4
    mha = MultiHeadAttention(d_model, num_heads)
    x = Tensor(
        np.random.default_rng(1).standard_normal((batch, seq_len, d_model)).astype(np.float32).ravel(),
        (batch, seq_len, d_model),
        requires_grad=True,
    )
    out = mha.forward(x, mask=causal_mask_tensor(seq_len))
    assert out.shape == (batch, seq_len, d_model)
    out.sum().backward()
    assert x.grad is not None


def test_rms_norm_forward_shape():
    from NimbleML.layers import RMSNorm

    ln = RMSNorm(8)
    x = Tensor(np.linspace(0.1, 0.8, 16, dtype=np.float64), (2, 8), requires_grad=True)
    out = ln.forward(x)
    assert out.shape == (2, 8)
    out.sum().backward()
    assert x.grad is not None
    assert ln.gamma.grad is not None


def test_gelu_forward_backward():
    x = Tensor(np.linspace(-1.0, 1.0, 4, dtype=np.float64), (4,), requires_grad=True)
    out = x.gelu()
    assert out.shape == (4,)
    out.sum().backward()
    assert x.grad is not None


def test_feedforward_forward_backward():
    from NimbleML.neural_network.feed_forward import FeedForward

    batch, seq_len, d_model = 2, 8, 32
    ff = FeedForward(d_model, ff_mult=4)
    x = Tensor(
        np.random.default_rng(2).standard_normal((batch, seq_len, d_model)).astype(np.float64).ravel(),
        (batch, seq_len, d_model),
        requires_grad=True,
    )
    out = ff.forward(x)
    assert out.shape == (batch, seq_len, d_model)
    out.sum().backward()
    assert x.grad is not None
    for p in ff.parameters():
        assert p.grad is not None


def test_gpt_forward_shape():
    vocab_size, d_model, num_heads, num_layers, max_seq_len = 50, 32, 4, 2, 8
    batch, seq_len = 2, 8
    model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
    input_ids = Tensor.from_int64(np.tile(np.arange(seq_len, dtype=np.int64), batch), (batch, seq_len))
    logits = model.forward(input_ids)
    assert logits.shape == (batch, seq_len, vocab_size)


def test_gpt_tied_weights_no_lm_head():
    model = GPT(64, 32, 4, 2, 8)
    assert not hasattr(model, "lm_head")
    names = [name for name, _ in named_parameters(model)]
    assert names.count("token_emb.weights") == 1
    assert not any(name.startswith("lm_head") for name in names)


def test_gpt_pos_encoding_prefix():
    pos_emb = Embedding(16, 8)
    out = pos_emb.forward_prefix(4)
    assert out.shape == (4, 8)
    out.sum().backward()
    grad = np.asarray(pos_emb.weights.grad).reshape(16, 8)
    assert np.allclose(grad[:4], np.ones((4, 8)))
    assert np.allclose(grad[4:], np.zeros((12, 8)))


def test_cross_entropy_3d_forward_backward():
    from NimbleML.losses import CrossEntropyLoss

    loss_fn = CrossEntropyLoss()
    logits = Tensor(
        np.random.default_rng(3).standard_normal((2, 3, 5)).astype(np.float64).ravel(),
        (2, 3, 5),
        requires_grad=True,
    )
    labels = Tensor.from_int64(np.array([1, 2, 0, 4, 3, 1], dtype=np.int64), (2, 3))
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    loss.backward()
    assert logits.grad is not None


def test_cross_entropy_ignore_index():
    from NimbleML.losses import CrossEntropyLoss

    loss_fn = CrossEntropyLoss()
    logits = Tensor(np.linspace(-1, 1, 12, dtype=np.float64), (2, 2, 3), requires_grad=True)
    labels = Tensor.from_int64(np.array([0, -1, 2, 1], dtype=np.int64), (2, 2))
    loss = loss_fn(logits, labels, ignore_index=-1)
    loss.backward()
    grad = np.asarray(logits.grad).reshape(2, 2, 3)
    assert np.allclose(grad[0, 1], np.zeros(3))
    assert np.any(grad != 0)


def test_gpt_checkpoint_save_load(tmp_path=None):
    ckpt_path = Path(__file__).parent / "_test_ckpt_gpt.npz" if tmp_path is None else tmp_path / "gpt.npz"
    try:
        vocab_size, d_model, num_heads, num_layers, max_seq_len = 40, 24, 4, 2, 8
        batch, seq_len = 2, 8
        model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
        input_ids = Tensor.from_int64(np.tile(np.arange(seq_len, dtype=np.int64), batch), (batch, seq_len))
        expected = np.asarray(model.forward(input_ids).data).reshape(batch, seq_len, vocab_size)
        save(model, ckpt_path)
        fresh = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
        load(fresh, ckpt_path)
        actual = np.asarray(fresh.forward(input_ids).data).reshape(batch, seq_len, vocab_size)
        assert np.allclose(expected, actual, atol=1e-5)
        assert not any(name.startswith("lm_head") for name, _ in named_parameters(fresh))
    finally:
        if tmp_path is None and ckpt_path.exists():
            ckpt_path.unlink()


def test_zero_grad_set_to_none():
    x = Tensor(np.array([1.0, 2.0], dtype=np.float64), (2,), requires_grad=True)
    x.zero_grad()
    assert x.grad is not None
    x.zero_grad(set_to_none=True)
    assert x.grad is None


def test_adamw_step():
    from NimbleML.optimizers import Adam, AdamW

    param = Tensor(np.array([1.0], dtype=np.float64), (1,), requires_grad=True)
    opt = AdamW([param], learning_rate=0.1, weight_decay=0.1)
    param.grad = np.array([0.5], dtype=np.float64)
    before = float(param.data[0])
    opt.step()
    after = float(param.data[0])
    assert after < before

    param2 = Tensor(np.array([1.0], dtype=np.float64), (1,), requires_grad=True)
    opt2 = Adam([param2], learning_rate=0.1)
    param2.grad = np.array([0.5], dtype=np.float64)
    before2 = float(param2.data[0])
    opt2.step()
    after2 = float(param2.data[0])
    assert after < after2


def test_adamw_param_groups_exclude_norm_and_bias():
    from toyGPT.train_utils import adamw_param_groups

    model = GPT(40, 24, 4, 2, 8)
    groups = adamw_param_groups(model, lr=3e-4, weight_decay=0.1)
    decay = groups[0]["params"]
    nodecay = groups[1]["params"]
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    assert len(decay) > 0
    assert len(nodecay) > 0
    decay_names = {name for name, p in named_parameters(model) if p in decay}
    assert all(name.endswith(".weights") for name in decay_names)
    nodecay_names = {name for name, p in named_parameters(model) if p in nodecay}
    assert all(not name.endswith(".weights") for name in nodecay_names)


def test_step_lr_scheduler():
    optimizer = SGD([Tensor([1.0], (1,), requires_grad=True)], learning_rate=1.0)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    scheduler.step()
    scheduler.step()
    scheduler.step()
    assert abs(optimizer.learning_rate - 0.1) < 1e-12


def test_clip_grad_norm_enforces_global_cap():
    p1 = Tensor([1.0, 2.0], (2,), requires_grad=True)
    p2 = Tensor([3.0, 4.0], (2,), requires_grad=True)
    p1.grad = np.array([30.0, 40.0], dtype=np.float64)
    p2.grad = np.array([50.0, 60.0], dtype=np.float64)
    original = _global_grad_norm([p1, p2])
    returned = clip_grad_norm_([p1, p2], max_norm=10.0)
    clipped = _global_grad_norm([p1, p2])
    assert returned == original
    # clip_grad_norm_ uses backend compute dtype (float32 by default), so allow
    # tiny numeric overshoot after scaling.
    assert clipped <= 10.0 + 1e-3


def test_checkpoint_save_load_dense(tmp_path=None):
    ckpt_path = Path(__file__).parent / "_test_ckpt_dense.npz" if tmp_path is None else tmp_path / "dense.npz"
    try:
        model = Dense(4, 2)
        model.weights.data = np.linspace(0.1, 0.8, 8, dtype=np.float64)
        model.biases.data = np.array([0.5, -0.5], dtype=np.float64)
        x = Tensor(np.linspace(1, 4, 4, dtype=np.float64), (1, 4))
        expected = np.asarray(model.forward(x).data).reshape(1, 2)
        save(model, ckpt_path)
        fresh = Dense(4, 2)
        load(fresh, ckpt_path)
        actual = np.asarray(fresh.forward(x).data).reshape(1, 2)
        assert np.allclose(expected, actual, atol=1e-6)
    finally:
        if tmp_path is None and ckpt_path.exists():
            ckpt_path.unlink()


def main():
    test_tensor_broadcast_backward()
    test_dense_forward_backward()
    test_conv2d_forward_backward()
    test_maxpool2d_backward_mask()
    test_embedding_backward_accumulates()
    test_softmax_normalization_and_grad()
    test_attention_shape_with_causal_mask()
    test_multi_head_attention_forward_backward()
    test_rms_norm_forward_shape()
    test_gelu_forward_backward()
    test_feedforward_forward_backward()
    test_gpt_forward_shape()
    test_gpt_tied_weights_no_lm_head()
    test_gpt_pos_encoding_prefix()
    test_cross_entropy_3d_forward_backward()
    test_cross_entropy_ignore_index()
    test_gpt_checkpoint_save_load()
    test_step_lr_scheduler()
    test_clip_grad_norm_enforces_global_cap()
    test_zero_grad_set_to_none()
    test_adamw_step()
    test_adamw_param_groups_exclude_norm_and_bias()
    test_checkpoint_save_load_dense()
    from tests.test_tokenizer import main as test_tokenizer_main

    test_tokenizer_main()
    print("Mandatory tests passed.")


if __name__ == "__main__":
    main()
