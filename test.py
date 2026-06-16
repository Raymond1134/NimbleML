from pathlib import Path

from NimbleML.utils.np_backend import set_dtype

# Pin float64 for the test suite: finite-difference gradchecks (tol=1e-3) and the
# many exact-value assertions (tol=1e-6) are too tight for float32 round-off.
set_dtype("float64")

from NimbleML.data.text import batch_sequences, build_vocab, decode, encode, load_text
from NimbleML.layers.conv2D import Conv2D, _im2col
from NimbleML.layers.dense import Dense
from NimbleML.layers.flatten import Flatten
from NimbleML.layers import Embedding, LayerNorm, MaxPool2D
from NimbleML.losses import CrossEntropyLoss
from NimbleML.optimizers import Adam, LRScheduler, StepLR, SGD
from NimbleML.activations import Softmax
from NimbleML.neural_network.attention import Attention, MultiHeadAttention, make_causal_mask
from NimbleML.neural_network.feed_forward import FeedForward
from NimbleML.neural_network.module import residual
from NimbleML.models.gpt import GPT
from NimbleML.neural_network.transformer import TransformerBlock
from NimbleML.utils.saveload import load, named_parameters, save
from NimbleML.utils.gradcheck import gradcheck
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def _assert_list_close(label, actual, expected, tol=1e-6):
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: length {len(actual)} != {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > tol:
            raise AssertionError(f"{label}[{i}]: {a} != {e}")


def test_forward_broadcasting():
    a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    b = Tensor([10, 20, 30], (3,))
    out = a + b
    _assert_list_close("(2,3)+(3,) data", out.data, [11, 22, 33, 14, 25, 36])

    a = Tensor([1, 2], (2, 1))
    b = Tensor([10, 20, 30], (1, 3))
    out = a + b
    _assert_list_close("(2,1)+(1,3) data", out.data, [11, 21, 31, 12, 22, 32])

    s = Tensor([2.0], ())
    m = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    out = s + m
    _assert_list_close("scalar+(2,3) data", out.data, [3, 4, 5, 6, 7, 8])

    out = m + s
    _assert_list_close("(2,3)+scalar data", out.data, [3, 4, 5, 6, 7, 8])


def test_backward_broadcasting():
    a = Tensor([1, 2], (2, 1), requires_grad=True)
    b = Tensor([10, 20, 30], (1, 3), requires_grad=True)
    loss = (a + b).sum()
    loss.backward()
    _assert_list_close("grad a (2,1)", a.grad, [3, 3])
    _assert_list_close("grad b (1,3)", b.grad, [2, 2, 2])

    a = Tensor([1, 2, 3], (3,), requires_grad=True)
    b = Tensor([10, 20, 30, 40, 50, 60], (2, 3), requires_grad=True)
    loss = (a + b).sum()
    loss.backward()
    _assert_list_close("grad a (3,)", a.grad, [2, 2, 2])
    _assert_list_close("grad b (2,3)", b.grad, [1, 1, 1, 1, 1, 1])


def test_im2col():
    x = np.arange(1, 17, dtype=np.float64).reshape(1, 1, 4, 4)

    cols, meta = _im2col(x, kernel_size=3, stride=1, padding=0)

    assert cols.shape == (4, 9), f"expected (4, 9), got {cols.shape}"
    assert meta["out_H"] == 2 and meta["out_W"] == 2
    assert meta["N"] == 1 and meta["C"] == 1 and meta["H"] == 4 and meta["W"] == 4

    expected_patches = [
        [1, 2, 3, 5, 6, 7, 9, 10, 11],
        [2, 3, 4, 6, 7, 8, 10, 11, 12],
        [5, 6, 7, 9, 10, 11, 13, 14, 15],
        [6, 7, 8, 10, 11, 12, 14, 15, 16],
    ]
    for i, expected in enumerate(expected_patches):
        _assert_list_close(f"im2col patch {i}", np.asarray(cols[i]), expected)


def test_conv2d_forward():
    layer = Conv2D(1, 2, kernel_size=3, stride=1, padding=0, bias=True)
    x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    out = layer.forward(x)
    assert out.shape == (1, 2, 2, 2), f"expected (1, 2, 2, 2), got {out.shape}"


def test_conv2d_backward():
    np.random.seed(0)
    layer = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
    x = Tensor(np.arange(1, 10, dtype=np.float64), (1, 1, 3, 3), requires_grad=True)
    out = layer.forward(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weights.grad is not None
    assert layer.biases.grad is not None
    assert len(x.grad) == 9
    assert len(layer.weights.grad) == 4
    assert len(layer.biases.grad) == 1


def test_maxpool2d_forward():
    layer = MaxPool2D(kernel_size=2, stride=2)
    x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    out = layer.forward(x)
    assert out.shape == (1, 1, 2, 2)
    _assert_list_close("maxpool values", np.asarray(out.data), [6, 8, 14, 16])


def test_maxpool2d_backward():
    layer = MaxPool2D(kernel_size=2, stride=2)
    x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    out = layer.forward(x)
    loss = out.sum()
    loss.backward()
    _assert_list_close("maxpool input grad", np.asarray(x.grad), [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1])


def test_flatten():
    layer = Flatten()
    x = Tensor(np.arange(24, dtype=np.float64), (2, 3, 2, 2), requires_grad=True)
    out = layer.forward(x)
    assert out.shape == (2, 12)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert len(x.grad) == 24
    _assert_list_close("flatten grad", np.asarray(x.grad), np.ones(24))


def test_embedding_forward_shape():
    layer = Embedding(vocab_size=10, embed_dim=4)
    layer.weights.data = np.arange(40, dtype=np.float64)
    ids = [[0, 2, 5], [1, 3, 9]]
    out = layer.forward(ids)
    assert out.shape == (2, 3, 4), f"expected (2, 3, 4), got {out.shape}"
    _assert_list_close("embedding row (0,0)", np.asarray(out.data.reshape(2, 3, 4)[0, 0]), [0, 1, 2, 3])
    _assert_list_close("embedding row (1,2)", np.asarray(out.data.reshape(2, 3, 4)[1, 2]), [36, 37, 38, 39])


def test_embedding_backward():
    layer = Embedding(vocab_size=10, embed_dim=4)
    ids = [[0, 2, 2], [1, 3, 0]]
    out = layer.forward(ids)
    loss = out.sum()
    loss.backward()
    assert layer.weights.grad is not None
    assert len(layer.weights.grad) == 40


def test_layernorm_output_mean_approx_0():
    layer = LayerNorm(4)
    x = Tensor(np.linspace(-2, 2, 24, dtype=np.float64), (2, 3, 4), requires_grad=True)
    out = layer.forward(x)
    last_dim_means = np.mean(out.data.reshape(2, 3, 4), axis=-1)
    assert np.allclose(last_dim_means, 0.0, atol=1e-5), f"expected ~0 means, got {last_dim_means}"


def test_layernorm_backward():
    layer = LayerNorm(4)
    layer.gamma.data = np.linspace(0.5, 1.5, 4, dtype=np.float64)
    layer.beta.data = np.linspace(-0.2, 0.2, 4, dtype=np.float64)
    x = Tensor(np.linspace(-1, 1, 24, dtype=np.float64), (2, 3, 4), requires_grad=True)
    tensors = [x, layer.gamma, layer.beta]

    def fn():
        for t in tensors:
            t.grad = None
        return layer.forward(x).sum()

    gradcheck(fn, tensors)


def test_matmul_2d():
    a = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), (2, 3), requires_grad=True)
    b = Tensor(np.arange(12, dtype=np.float64).reshape(3, 4), (3, 4), requires_grad=True)
    out = a @ b
    assert out.shape == (2, 4)
    out.sum().backward()
    assert a.grad is not None and b.grad is not None


def test_matmul_3d():
    a = Tensor(np.linspace(0, 1, 80, dtype=np.float64), (2, 5, 8), requires_grad=True)
    b = Tensor(np.linspace(0, 1, 32, dtype=np.float64), (8, 4), requires_grad=True)
    out = a @ b
    assert out.shape == (2, 5, 4), f"expected (2, 5, 4), got {out.shape}"
    out.sum().backward()
    assert a.grad is not None and b.grad is not None


def test_gradcheck_matmul_2d():
    a = Tensor(np.linspace(0, 1, 6, dtype=np.float64), (2, 3), requires_grad=True)
    b = Tensor(np.linspace(0, 1, 12, dtype=np.float64), (3, 4), requires_grad=True)
    tensors = [a, b]

    def fn():
        for t in tensors:
            t.grad = None
        return (a @ b).sum()

    gradcheck(fn, tensors)


def test_gradcheck_matmul_3d():
    a = Tensor(np.linspace(0, 1, 80, dtype=np.float64), (2, 5, 8), requires_grad=True)
    b = Tensor(np.linspace(0, 1, 32, dtype=np.float64), (8, 4), requires_grad=True)
    tensors = [a, b]

    def fn():
        for t in tensors:
            t.grad = None
        return (a @ b).sum()

    gradcheck(fn, tensors)


def test_dense_3d():
    layer = Dense(8, 4)
    x2 = Tensor(np.linspace(0, 1, 16, dtype=np.float64), (2, 8), requires_grad=True)
    out2 = layer.forward(x2)
    assert out2.shape == (2, 4)

    x3 = Tensor(np.linspace(0, 1, 80, dtype=np.float64), (2, 5, 8), requires_grad=True)
    out3 = layer.forward(x3)
    assert out3.shape == (2, 5, 4)
    out3.sum().backward()
    assert x3.grad is not None and layer.weights.grad is not None


def test_text_encode_decode_roundtrip():
    text = "abc\n123!?"
    char_to_idx, idx_to_char = build_vocab(text)
    ids = encode(text, char_to_idx)
    assert decode(ids, idx_to_char) == text


def test_batch_sequences():
    ids = list(range(30))
    batch_size = 2
    seq_len = 4
    inputs, targets = next(batch_sequences(ids, batch_size, seq_len))
    assert inputs.shape == (batch_size, seq_len)
    assert targets.shape == (batch_size, seq_len)
    for row in range(batch_size):
        assert np.allclose(
            np.asarray(targets.data).reshape(batch_size, seq_len)[row],
            np.asarray(inputs.data).reshape(batch_size, seq_len)[row] + 1,
        )

    ids_arr = np.arange(30, dtype=np.int32)
    inputs, targets = next(batch_sequences(ids_arr, batch_size, seq_len))
    assert inputs.shape == (batch_size, seq_len)
    assert targets.shape == (batch_size, seq_len)


def test_sequence_cross_entropy():
    logits = Tensor(np.linspace(0, 1, 60, dtype=np.float64), (2, 3, 10), requires_grad=True)
    targets = Tensor([0, 1, 2, 3, 4, 5], (2, 3))
    loss = CrossEntropyLoss()(logits, targets)
    assert loss.shape == ()
    loss.backward()
    assert logits.grad is not None


def test_load_text():
    path = Path(__file__).resolve().parent / "NimbleML" / "data" / "tiny_corpus.txt"
    ids, char_to_idx, idx_to_char = load_text(path)
    assert len(ids) > 0
    assert len(char_to_idx) == len(idx_to_char)
    assert decode(ids[:100], idx_to_char) == path.read_text(encoding="utf-8")[:100]


def test_softmax_3d():
    logits = Tensor(
        [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
        (2, 3, 3),
        requires_grad=True,
    )
    probs = Softmax()(logits)
    assert probs.shape == (2, 3, 3)
    row_sums = np.asarray(probs.data).reshape(2, 3, 3).sum(axis=-1)
    assert np.allclose(row_sums, 1.0)
    probs.sum().backward()
    assert logits.grad is not None


def test_causal_mask():
    mask = make_causal_mask(4)
    assert mask[0, 1] == -np.inf
    assert mask[1, 1] == 0
    assert mask[2, 0] == 0
    assert mask[1, 2] == -np.inf


def test_attention_forward_shape():
    batch, seq_len, d_k = 2, 4, 8
    rng = np.random.default_rng(0)
    Q = Tensor(rng.standard_normal((batch, seq_len, d_k)).ravel(), (batch, seq_len, d_k))
    K = Tensor(rng.standard_normal((batch, seq_len, d_k)).ravel(), (batch, seq_len, d_k))
    V = Tensor(rng.standard_normal((batch, seq_len, d_k)).ravel(), (batch, seq_len, d_k))
    out = Attention(d_k).forward(Q, K, V, mask=make_causal_mask(seq_len))
    assert out.shape == (batch, seq_len, d_k)


def test_gradcheck_attention():
    batch, seq_len, d_k = 1, 3, 4
    rng = np.random.default_rng(0)
    Q = Tensor(
        (rng.standard_normal((batch, seq_len, d_k)) * 0.1).ravel(),
        (batch, seq_len, d_k),
        requires_grad=True,
    )
    K = Tensor(
        (rng.standard_normal((batch, seq_len, d_k)) * 0.1).ravel(),
        (batch, seq_len, d_k),
        requires_grad=True,
    )
    V = Tensor(
        (rng.standard_normal((batch, seq_len, d_k)) * 0.1).ravel(),
        (batch, seq_len, d_k),
        requires_grad=True,
    )
    attn = Attention(d_k)
    tensors = [Q, K, V]

    def fn():
        for t in tensors:
            t.grad = None
        return attn.forward(Q, K, V).sum()

    gradcheck(fn, tensors)


def test_multi_head_attention_forward_shape():
    batch, seq_len, d_model, num_heads = 2, 4, 32, 4
    rng = np.random.default_rng(1)
    x = Tensor(rng.standard_normal((batch, seq_len, d_model)).ravel(), (batch, seq_len, d_model))
    out = MultiHeadAttention(d_model, num_heads).forward(x, mask=make_causal_mask(seq_len))
    assert out.shape == (batch, seq_len, d_model)


def test_feed_forward_shape():
    batch, seq_len, d_model = 2, 8, 32
    rng = np.random.default_rng(2)
    x = Tensor(rng.standard_normal((batch, seq_len, d_model)).ravel(), (batch, seq_len, d_model))
    out = FeedForward(d_model).forward(x)
    assert out.shape == (batch, seq_len, d_model)


def test_residual():
    x = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2), requires_grad=True)

    def scale(t):
        return t * 2.0

    out = residual(x, scale)
    assert out.shape == (2, 2)
    assert np.allclose(np.asarray(out.data).reshape(2, 2), 3.0 * np.asarray(x.data).reshape(2, 2))
    out.sum().backward()
    assert x.grad is not None


def test_transformer_block_shape():
    batch, seq_len, d_model, num_heads = 2, 8, 32, 4
    rng = np.random.default_rng(3)
    x = Tensor(rng.standard_normal((batch, seq_len, d_model)).ravel(), (batch, seq_len, d_model))
    out = TransformerBlock(d_model, num_heads).forward(x)
    assert out.shape == (batch, seq_len, d_model)


def test_gpt_forward_shape():
    vocab_size, d_model, num_heads, num_layers, max_seq_len = 50, 32, 4, 2, 8
    batch, seq_len = 2, 8
    model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
    input_ids = Tensor(
        np.tile(np.arange(seq_len, dtype=np.float64), batch),
        (batch, seq_len),
    )
    logits = model.forward(input_ids)
    assert logits.shape == (batch, seq_len, vocab_size)


def _host_array(tensor):
    arr = tensor.data.reshape(tensor.shape)
    return np.asarray(arr.get() if hasattr(arr, "get") else arr)


def test_lr_scheduler_base():
    layer = Dense(2, 1)
    optimizer = Adam(layer.parameters(), learning_rate=0.1)

    class DecayLR(LRScheduler):
        def get_lr(self):
            return [base * (0.5**self.last_epoch) for base in self.base_lrs]

    scheduler = DecayLR(optimizer)
    scheduler.step()
    assert scheduler.last_epoch == 0
    assert abs(optimizer.learning_rate - 0.1) < 1e-9
    scheduler.step()
    assert abs(optimizer.learning_rate - 0.05) < 1e-9
    scheduler.step(epoch=3)
    assert scheduler.last_epoch == 3
    assert abs(optimizer.learning_rate - 0.0125) < 1e-9


def test_step_lr():
    optimizer = SGD([Tensor([1.0], (1,), requires_grad=True)], learning_rate=1.0)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    scheduler.step()  # epoch 0
    assert abs(optimizer.learning_rate - 1.0) < 1e-9
    scheduler.step()  # epoch 1
    assert abs(optimizer.learning_rate - 1.0) < 1e-9
    scheduler.step()  # epoch 2
    assert abs(optimizer.learning_rate - 0.1) < 1e-9


def test_lr_scheduler_param_groups():
    w1 = Tensor([1.0], (1,), requires_grad=True)
    w2 = Tensor([2.0], (1,), requires_grad=True)
    optimizer = Adam(
        [
            {"params": [w1], "lr": 0.1},
            {"params": [w2], "lr": 0.2},
        ]
    )

    class HalfLR(LRScheduler):
        def get_lr(self):
            return [base * 0.5 for base in self.base_lrs]

    scheduler = HalfLR(optimizer)
    scheduler.step()
    assert optimizer.get_lr() == [0.05, 0.1]
    assert optimizer.param_groups[0]["lr"] == 0.05
    assert optimizer.param_groups[1]["lr"] == 0.1


def test_named_parameters_dense():
    layer = Dense(3, 2)
    names = [name for name, _ in named_parameters(layer)]
    assert names == ["weights", "biases"]


def test_checkpoint_save_load_dense(tmp_path=None):
    path = Path(__file__).parent / "_test_ckpt_dense.npz" if tmp_path is None else tmp_path / "dense.npz"
    try:
        model = Dense(4, 2)
        model.weights.data = np.linspace(0.1, 0.8, 8, dtype=np.float64)
        model.biases.data = np.array([0.5, -0.5], dtype=np.float64)

        x = Tensor(np.linspace(1, 4, 4, dtype=np.float64), (1, 4))
        expected = _host_array(model.forward(x))

        save(model, path)

        fresh = Dense(4, 2)
        load(fresh, path)
        actual = _host_array(fresh.forward(x))

        if not np.allclose(expected, actual, atol=1e-6):
            raise AssertionError(f"checkpoint reload mismatch: {expected} vs {actual}")
    finally:
        if tmp_path is None and path.exists():
            path.unlink()


def test_checkpoint_save_load_gpt(tmp_path=None):
    path = Path(__file__).parent / "_test_ckpt_gpt.npz" if tmp_path is None else tmp_path / "gpt.npz"
    vocab_size, d_model, num_heads, num_layers, max_seq_len = 20, 16, 4, 2, 4
    batch, seq_len = 2, 4
    try:
        model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
        input_ids = Tensor(
            np.tile(np.arange(seq_len, dtype=np.float64), batch),
            (batch, seq_len),
        )
        expected = _host_array(model.forward(input_ids))

        save(model, path)

        fresh = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
        load(fresh, path)
        actual = _host_array(fresh.forward(input_ids))

        if not np.allclose(expected, actual, atol=1e-6):
            raise AssertionError("GPT checkpoint reload mismatch")
    finally:
        if tmp_path is None and path.exists():
            path.unlink()


def test_gradcheck_dense():
    layer = Dense(2, 1)
    layer.weights.data = np.array([0.5, -0.3], dtype=np.float64)
    layer.biases.data = np.array([0.1], dtype=np.float64)
    x = Tensor([1.0, 2.0], (1, 2), requires_grad=True)
    tensors = [x, layer.weights, layer.biases]

    def fn():
        for t in tensors:
            t.grad = None
        return layer.forward(x).sum()

    gradcheck(fn, tensors)


def test_gradcheck_conv2d():
    layer = Conv2D(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
    layer.weights.data = np.linspace(0.1, 0.9, 9, dtype=np.float64)
    layer.biases.data = np.array([0.05], dtype=np.float64)
    x = Tensor(np.linspace(1, 16, 16, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    tensors = [x, layer.weights, layer.biases]

    def fn():
        for t in tensors:
            t.grad = None
        return layer.forward(x).sum()

    gradcheck(fn, tensors)


def test_gradcheck_maxpool2d():
    layer = MaxPool2D(kernel_size=2, stride=2)
    x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
    tensors = [x]

    def fn():
        for t in tensors:
            t.grad = None
        return layer.forward(x).sum()

    gradcheck(fn, tensors)


def main():
    test_forward_broadcasting()
    test_backward_broadcasting()
    test_im2col()
    test_conv2d_forward()
    test_conv2d_backward()
    test_maxpool2d_forward()
    test_maxpool2d_backward()
    test_flatten()
    test_embedding_forward_shape()
    test_embedding_backward()
    test_layernorm_output_mean_approx_0()
    test_layernorm_backward()
    test_matmul_2d()
    test_matmul_3d()
    test_gradcheck_matmul_2d()
    test_gradcheck_matmul_3d()
    test_dense_3d()
    test_text_encode_decode_roundtrip()
    test_sequence_cross_entropy()
    test_load_text()
    test_batch_sequences()
    test_softmax_3d()
    test_causal_mask()
    test_attention_forward_shape()
    test_gradcheck_attention()
    test_multi_head_attention_forward_shape()
    test_feed_forward_shape()
    test_residual()
    test_transformer_block_shape()
    test_gpt_forward_shape()
    test_lr_scheduler_base()
    test_step_lr()
    test_lr_scheduler_param_groups()
    test_named_parameters_dense()
    test_checkpoint_save_load_dense()
    test_checkpoint_save_load_gpt()
    test_gradcheck_dense()
    test_gradcheck_conv2d()
    test_gradcheck_maxpool2d()
    print("All tests passed.")


if __name__ == "__main__":
    main()
