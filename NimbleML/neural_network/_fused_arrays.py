"""Array-level fused ops for transformer blocks (no intermediate Tensor nodes)."""
from __future__ import annotations
from math import prod
from NimbleML.kernels.fused_rmsnorm import fused_rmsnorm_backward, fused_rmsnorm_forward
from NimbleML.kernels.fused_sdpa import fused_sdpa_backward, fused_sdpa_forward
from NimbleML.utils.activations import gelu_backward, gelu_forward
from NimbleML.utils.np_backend import np
from NimbleML.utils.rope import apply_rope_flat


def _as2d(x, d_last):
    row = prod(x.shape[:-1]) if x.ndim > 1 else 1
    return x.reshape(row, d_last), row


def _tensor_param(tensor):
    return tensor._view()


def _t(w):
    """Transposed view for GEMM operands — cuBLAS/BLAS handle strides, no copy."""
    return np.swapaxes(w, -2, -1)


def _dense_forward(x2d, weight, bias=None):
    w = _tensor_param(weight)
    out = x2d @ w
    if bias is not None:
        out = out + _tensor_param(bias).reshape(-1)
    return out, x2d, w


def _dense_backward(grad_out, x2d, w, *, weight, bias=None):
    if weight.requires_grad:
        grad_w = np.matmul(_t(x2d), grad_out)
        weight._accumulate_grad(grad_w.ravel())
    if bias is not None and bias.requires_grad:
        bias._accumulate_grad(np.sum(grad_out, axis=0).ravel())
    return np.matmul(grad_out, _t(w))


def _split_heads_array(arr, batch, seq_len, num_heads, d_k):
    shape_4d = arr.reshape(batch, seq_len, num_heads, d_k)
    return np.transpose(shape_4d, (0, 2, 1, 3)).reshape(batch * num_heads, seq_len, d_k)


def _merge_heads_array(arr, batch, seq_len, num_heads, d_k):
    shape_4d = arr.reshape(batch, num_heads, seq_len, d_k)
    return np.transpose(shape_4d, (0, 2, 1, 3)).reshape(batch, seq_len, num_heads * d_k)


def mha_forward_arrays(x_arr, mha, mask_arr):
    """Multi-head self-attention on ``(batch, seq, d_model)`` activations.

    Q/K/V projections run as one fused GEMM against the concatenated weight
    matrix. When the attention module carries a RoPE cache (set by ``GPT``
    when ``use_rope=True``), rotary embeddings are applied to Q and K after
    the head split.
    """
    batch, seq_len, d_model = x_arr.shape
    num_heads = mha.num_heads
    d_k = mha.d_k
    scale = mha.scale

    x2d, _ = _as2d(x_arr, d_model)
    wq = _tensor_param(mha.W_q.weights)
    wk = _tensor_param(mha.W_k.weights)
    wv = _tensor_param(mha.W_v.weights)
    w_qkv = np.concatenate([wq, wk, wv], axis=1)  # (d_model, 3*d_model)

    qkv = x2d @ w_qkv
    biases = (mha.W_q.biases, mha.W_k.biases, mha.W_v.biases)
    has_bias = any(b is not None for b in biases)
    if has_bias:
        parts = [
            _tensor_param(b).reshape(-1)
            if b is not None
            else np.zeros(d_model, dtype=qkv.dtype)
            for b in biases
        ]
        qkv = qkv + np.concatenate(parts)

    q = _split_heads_array(
        qkv[:, :d_model].reshape(batch, seq_len, d_model), batch, seq_len, num_heads, d_k
    )
    k = _split_heads_array(
        qkv[:, d_model : 2 * d_model].reshape(batch, seq_len, d_model),
        batch, seq_len, num_heads, d_k,
    )
    v = _split_heads_array(
        qkv[:, 2 * d_model :].reshape(batch, seq_len, d_model),
        batch, seq_len, num_heads, d_k,
    )

    rope = getattr(mha, "_rope_cache", None)
    if rope is not None:
        q = apply_rope_flat(q, rope[0], rope[1])
        k = apply_rope_flat(k, rope[0], rope[1])

    attn_out, probs = fused_sdpa_forward(
        q, k, v, scale, mask_arr, batch=batch, num_heads=num_heads
    )
    merged = _merge_heads_array(attn_out, batch, seq_len, num_heads, d_k)
    merged2d, _ = _as2d(merged, d_model)
    out2d, save_merged, wo = _dense_forward(merged2d, mha.W_o.weights, mha.W_o.biases)
    out = out2d.reshape(batch, seq_len, d_model)

    ctx = {
        "batch": batch,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_k": d_k,
        "scale": scale,
        "x2d": x2d,
        "w_qkv": w_qkv,
        "wo": wo,
        "q": q,  # rotated when rope is active — what SDPA backward needs
        "k": k,
        "v": v,
        "probs": probs,
        "save_merged": save_merged,
        "rope": rope,
        "mha": mha,
    }
    return out, ctx


def mha_backward_arrays(grad_out, ctx):
    mha = ctx["mha"]
    batch = ctx["batch"]
    seq_len = ctx["seq_len"]
    d_model = ctx["d_model"]
    num_heads = ctx["num_heads"]
    d_k = ctx["d_k"]
    rope = ctx["rope"]

    grad2d, _ = _as2d(grad_out, d_model)
    grad_merged2d = np.matmul(grad2d, _t(ctx["wo"]))
    if mha.W_o.weights.requires_grad:
        grad_wo = np.matmul(_t(ctx["save_merged"]), grad2d)
        mha.W_o.weights._accumulate_grad(grad_wo.ravel())
    if mha.W_o.biases is not None and mha.W_o.biases.requires_grad:
        mha.W_o.biases._accumulate_grad(np.sum(grad2d, axis=0).ravel())

    grad_attn = grad_merged2d.reshape(batch, seq_len, d_model)
    grad_attn_heads = _split_heads_array(grad_attn, batch, seq_len, num_heads, d_k)
    grad_q, grad_k, grad_v = fused_sdpa_backward(
        grad_attn_heads,
        ctx["q"],
        ctx["k"],
        ctx["v"],
        ctx["probs"],
        ctx["scale"],
    )

    if rope is not None:
        # q_rot = R(q): rotations are orthogonal, so dq = R^T(dq_rot) = R(-theta).
        grad_q = apply_rope_flat(grad_q, rope[0], rope[1], inverse=True)
        grad_k = apply_rope_flat(grad_k, rope[0], rope[1], inverse=True)

    grad_q2d = _merge_heads_array(grad_q, batch, seq_len, num_heads, d_k).reshape(batch * seq_len, d_model)
    grad_k2d = _merge_heads_array(grad_k, batch, seq_len, num_heads, d_k).reshape(batch * seq_len, d_model)
    grad_v2d = _merge_heads_array(grad_v, batch, seq_len, num_heads, d_k).reshape(batch * seq_len, d_model)
    grad_qkv = np.concatenate([grad_q2d, grad_k2d, grad_v2d], axis=1)  # (rows, 3d)

    weights = (mha.W_q.weights, mha.W_k.weights, mha.W_v.weights)
    if any(w.requires_grad for w in weights):
        grad_w_qkv = np.matmul(_t(ctx["x2d"]), grad_qkv)  # (d, 3d) — one GEMM
        for i, w in enumerate(weights):
            if w.requires_grad:
                w._accumulate_grad(
                    grad_w_qkv[:, i * d_model : (i + 1) * d_model].ravel()
                )

    biases = (mha.W_q.biases, mha.W_k.biases, mha.W_v.biases)
    if any(b is not None and b.requires_grad for b in biases):
        grad_b_qkv = np.sum(grad_qkv, axis=0)
        for i, b in enumerate(biases):
            if b is not None and b.requires_grad:
                b._accumulate_grad(grad_b_qkv[i * d_model : (i + 1) * d_model].ravel())

    grad_x = np.matmul(grad_qkv, _t(ctx["w_qkv"]))  # (rows, d) — one GEMM
    return grad_x.reshape(batch, seq_len, d_model)


def ffn_forward_arrays(x_arr, ffn):
    in_shape = x_arr.shape
    d_in = in_shape[-1]
    d_hidden = ffn.dense1.weights.shape[1]
    d_out = ffn.dense2.weights.shape[1]
    row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1

    x2d = x_arr.reshape(row_count, d_in)
    w1 = _tensor_param(ffn.dense1.weights)
    w2 = _tensor_param(ffn.dense2.weights)
    b1 = _tensor_param(ffn.dense1.biases) if ffn.dense1.biases is not None else None
    b2 = _tensor_param(ffn.dense2.biases) if ffn.dense2.biases is not None else None

    pre_act = x2d @ w1
    if b1 is not None:
        pre_act = pre_act + b1
    hidden, tanh_u = gelu_forward(pre_act)
    out2d = hidden @ w2
    if b2 is not None:
        out2d = out2d + b2

    ctx = {
        "ffn": ffn,
        "in_shape": in_shape,
        "row_count": row_count,
        "d_in": d_in,
        "d_hidden": d_hidden,
        "d_out": d_out,
        "save_x": x2d,
        "save_pre": pre_act,
        "save_tanh_u": tanh_u,
        "save_hidden": hidden,
        "w1": w1,
        "w2": w2,
    }
    return out2d.reshape(in_shape), ctx


def ffn_backward_arrays(grad_out, ctx):
    ffn = ctx["ffn"]
    row_count = ctx["row_count"]
    d_out = ctx["d_out"]
    in_shape = ctx["in_shape"]

    grad_out2d = grad_out.reshape(row_count, d_out)

    if ffn.dense2.weights.requires_grad:
        grad_w2 = np.matmul(_t(ctx["save_hidden"]), grad_out2d)
        ffn.dense2.weights._accumulate_grad(grad_w2.ravel())
    if ffn.dense2.biases is not None and ffn.dense2.biases.requires_grad:
        ffn.dense2.biases._accumulate_grad(np.sum(grad_out2d, axis=0).ravel())

    grad_hidden = np.matmul(grad_out2d, _t(ctx["w2"]))
    grad_pre = gelu_backward(grad_hidden, ctx["save_pre"], ctx["save_tanh_u"])

    if ffn.dense1.weights.requires_grad:
        grad_w1 = np.matmul(_t(ctx["save_x"]), grad_pre)
        ffn.dense1.weights._accumulate_grad(grad_w1.ravel())
    if ffn.dense1.biases is not None and ffn.dense1.biases.requires_grad:
        ffn.dense1.biases._accumulate_grad(np.sum(grad_pre, axis=0).ravel())

    return np.matmul(grad_pre, _t(ctx["w1"])).reshape(in_shape)


def fused_block_forward_arrays(x_arr, block, mask_arr):
    """Pre-norm block: x + MHA(LN1(x)); x + FFN(LN2(x))."""
    shape = x_arr.shape
    d = shape[-1]
    row_count = prod(shape[:-1]) if len(shape) > 1 else 1
    g1 = _tensor_param(block.ln1.gamma)
    norm1, save_x1, ms1, rms1 = fused_rmsnorm_forward(x_arr.reshape(row_count, d), g1, block.ln1.epsilon)
    norm1 = norm1.reshape(x_arr.shape)

    mha_out, mha_ctx = mha_forward_arrays(norm1, block.mha, mask_arr)
    x_mid = x_arr + mha_out

    g2 = _tensor_param(block.ln2.gamma)
    row_mid, _ = _as2d(x_mid, x_mid.shape[-1])
    norm2, save_x2, ms2, rms2 = fused_rmsnorm_forward(row_mid, g2, block.ln2.epsilon)
    norm2 = norm2.reshape(x_mid.shape)

    ffn_out, ffn_ctx = ffn_forward_arrays(norm2, block.ffn)
    out = x_mid + ffn_out

    ctx = {
        "block": block,
        "mask_arr": mask_arr,
        "x0": x_arr,
        "x_mid": x_mid,
        "ln1": (save_x1, g1, ms1, rms1, block.ln1.epsilon),
        "ln2": (save_x2, g2, ms2, rms2, block.ln2.epsilon),
        "mha_ctx": mha_ctx,
        "ffn_ctx": ffn_ctx,
        "shape": shape,
        "d": d,
        "row_count": row_count,
    }
    return out, ctx


def fused_trunk_forward_arrays(x_arr, blocks, ln, mask_arr):
    """Run ``N`` fused blocks and final RMSNorm on arrays."""
    ctxs = []
    cur = x_arr
    for block in blocks:
        cur, ctx = fused_block_forward_arrays(cur, block, mask_arr)
        ctxs.append(ctx)
    row_count = prod(cur.shape[:-1]) if cur.ndim > 1 else 1
    d = cur.shape[-1]
    g_ln = _tensor_param(ln.gamma)
    out, save_x, ms, rms = fused_rmsnorm_forward(cur.reshape(row_count, d), g_ln, ln.epsilon)
    out = out.reshape(cur.shape)
    trunk_ctx = {
        "blocks": blocks,
        "ln": ln,
        "block_ctxs": ctxs,
        "ln_ctx": (save_x, g_ln, ms, rms, ln.epsilon),
        "shape": cur.shape,
        "d": d,
        "row_count": row_count,
    }
    return out, trunk_ctx


def fused_trunk_backward_arrays(grad_out, trunk_ctx):
    ln = trunk_ctx["ln"]
    save_x, g_ln, ms, rms, epsilon = trunk_ctx["ln_ctx"]
    shape = trunk_ctx["shape"]
    d = trunk_ctx["d"]
    row_count = trunk_ctx["row_count"]

    grad_cur, grad_g_ln = fused_rmsnorm_backward(
        grad_out.reshape(row_count, d),
        save_x,
        g_ln,
        ms,
        rms,
        epsilon,
    )
    grad_cur = grad_cur.reshape(shape)
    if ln.gamma.requires_grad:
        ln.gamma._accumulate_grad(grad_g_ln.ravel())

    for ctx in reversed(trunk_ctx["block_ctxs"]):
        grad_cur = fused_block_backward_arrays(grad_cur, ctx)
    return grad_cur


def fused_block_backward_arrays(grad_out, ctx):
    block = ctx["block"]
    epsilon1 = ctx["ln1"][4]
    epsilon2 = ctx["ln2"][4]
    shape = ctx["shape"]
    d = ctx["d"]
    row_count = ctx["row_count"]

    grad_mid = grad_out.copy()
    grad_ffn = grad_out
    grad_ffn_in = ffn_backward_arrays(grad_ffn, ctx["ffn_ctx"])

    save_x2, g2, ms2, rms2, _ = ctx["ln2"]
    grad_norm2, grad_g2 = fused_rmsnorm_backward(
        grad_ffn_in.reshape(row_count, d),
        save_x2,
        g2,
        ms2,
        rms2,
        epsilon2,
    )
    grad_mid += grad_norm2.reshape(shape)
    if block.ln2.gamma.requires_grad:
        block.ln2.gamma._accumulate_grad(grad_g2.ravel())

    grad_x0 = grad_mid.copy()
    grad_mha = grad_mid
    grad_mha_in = mha_backward_arrays(grad_mha, ctx["mha_ctx"])

    save_x1, g1, ms1, rms1, _ = ctx["ln1"]
    grad_norm1, grad_g1 = fused_rmsnorm_backward(
        grad_mha_in.reshape(row_count, d),
        save_x1,
        g1,
        ms1,
        rms1,
        epsilon1,
    )
    grad_x0 += grad_norm1.reshape(shape)
    if block.ln1.gamma.requires_grad:
        block.ln1.gamma._accumulate_grad(grad_g1.ravel())

    return grad_x0
