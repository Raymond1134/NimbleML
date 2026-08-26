"""Chat / instruction datasets with assistant-only loss masks."""
from __future__ import annotations

import re
from NimbleML.data.dataset import Dataset, PADDED_LABEL
from NimbleML.utils.np_backend import np

SYSTEM_PERSONA = (
    "You're Nimble, a casual STEM-loving friend. Talk like a sharp person, not a "
    "corporate assistant. Think privately in a short <think>...</think> block first "
    "(plan, units, pitfalls), then answer. Match reply length to the user: one-liners "
    "for small talk, worked steps for real problems. Don't lecture. If you're unsure, "
    "say so. Prefer SI units and real math; skip fluff."
)

DEFAULT_THINK_PREFIX = (
    "<think>\n"
    "Goal: answer this clearly. Note the key fact, units, or steps, then reply at "
    "the right length.\n"
    "</think>\n"
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)

# Plain-text ChatML markers (BPE pieces, not extra special tokens).
CHATML_STOP_STRINGS = ("<|end|>", "<|user|>", "<|system|>", "<|assistant|>")


def encode_token_ids(tokenizer, text: str) -> list[int]:
    """Return token ids from ``tokenizer.encode`` (list or HF Encoding)."""
    enc = tokenizer.encode(text)
    if hasattr(enc, "ids"):
        return [int(i) for i in enc.ids]
    return [int(i) for i in list(enc)]


def strip_think(text: str) -> str:
    """Remove a leading/embedded ``<think>...</think>`` block from assistant text."""
    out = _THINK_RE.sub("", text or "", count=1)
    # Half-trained SFT often emits an unclosed <think> or a broken <|endthink|>.
    cut_at = None
    lower = out.lower()
    for marker in ("<think>", "</think>", "<|endthink|>", "<|endthink>"):
        idx = lower.find(marker)
        if idx >= 0:
            cut_at = idx if cut_at is None else min(cut_at, idx)
    if cut_at is not None:
        out = out[:cut_at]
    return out.lstrip()


def wrap_assistant_content(content: str) -> str:
    """Ensure assistant text starts with a short think block (template, not a model)."""
    text = content or ""
    if _THINK_RE.search(text):
        return text
    return DEFAULT_THINK_PREFIX + text


def ensure_system_message(messages: list[dict], persona: str = SYSTEM_PERSONA) -> list[dict]:
    """Prepend the STEM-friend system turn when missing."""
    if messages and messages[0].get("role") == "system":
        out = [dict(messages[0])]
        if not (out[0].get("content") or "").strip():
            out[0]["content"] = persona
        out.extend(dict(m) for m in messages[1:])
        return out
    return [{"role": "system", "content": persona}, *(dict(m) for m in messages)]


class ChatSFTDataset(Dataset):
    """Tokenized chat examples with labels masked on non-assistant tokens.

    Each item is a dict::

        {"input_ids": [...], "labels": [...]}  # labels use PADDED_LABEL where ignored
    """

    def __init__(self, examples: list[dict], *, max_seq_len: int):
        self.examples = examples
        self.max_seq_len = int(max_seq_len)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = list(ex["input_ids"])[: self.max_seq_len]
        labels = list(ex["labels"])[: self.max_seq_len]
        return {"input_ids": ids, "labels": labels}


def collate_chat_batch(batch, *, pad_id: int = 0):
    """Pad a list of chat examples to a rectangular batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)
    inputs = np.full((bsz, max_len), pad_id, dtype=np.int64)
    labels = np.full((bsz, max_len), PADDED_LABEL, dtype=np.int64)
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        inputs[i, :n] = np.asarray(ex["input_ids"], dtype=np.int64)
        labels[i, :n] = np.asarray(ex["labels"], dtype=np.int64)
    return inputs, labels


def apply_chat_template(messages: list[dict], tokenizer, *, add_generation_prompt: bool = False):
    """Simple ChatML-style template.

    messages: list of {role, content} with roles in {system, user, assistant}.
    Returns token id list (no labels).
    """
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|{role}|>\n{content}<|end|>\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    text = "".join(parts)
    if hasattr(tokenizer, "encode"):
        return encode_token_ids(tokenizer, text)
    raise TypeError("tokenizer must provide encode()")


def build_sft_example(messages: list[dict], tokenizer, *, max_seq_len: int):
    """Build next-token SFT tensors; loss only on assistant content + end tokens.

    ``labels[t]`` is ``input_ids[t+1]`` (causal shift). The ``<|assistant|>`` header
    is masked; user/system spans are masked. Empty results have empty lists.
    """
    max_seq_len = int(max_seq_len)
    ids: list[int] = []
    is_target: list[bool] = []
    for i, m in enumerate(messages):
        role = m.get("role") or "user"
        if role == "assistant":
            header = apply_chat_template(messages[:i], tokenizer, add_generation_prompt=True)
            full = apply_chat_template(messages[: i + 1], tokenizer, add_generation_prompt=False)
            if len(header) <= len(full) and full[: len(header)] == header:
                h_new = header[len(ids) :]
                c_new = full[len(header) :]
            else:
                h_new = []
                c_new = full[len(ids) :]
            ids.extend(h_new)
            is_target.extend([False] * len(h_new))
            ids.extend(c_new)
            is_target.extend([True] * len(c_new))
        else:
            full = apply_chat_template(messages[: i + 1], tokenizer, add_generation_prompt=False)
            new = full[len(ids) :]
            ids.extend(new)
            is_target.extend([False] * len(new))

    cap = max_seq_len + 1
    ids = ids[:cap]
    is_target = is_target[:cap]
    if len(ids) < 2:
        return {"input_ids": [], "labels": []}
    input_ids = ids[:-1]
    labels = [
        int(ids[t + 1]) if is_target[t + 1] else PADDED_LABEL for t in range(len(input_ids))
    ]
    return {"input_ids": input_ids, "labels": labels}


def pack_sft_sequences(
    examples: list[dict],
    *,
    seq_len: int,
    pad_id: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pack variable-length SFT examples into fixed ``seq_len`` rows.

    Does not cut inside an example except when a single example exceeds ``seq_len``
    (then it is truncated). Remainder of a row is padded with ``pad_id`` / ignore.
    """
    seq_len = int(seq_len)
    rows_x: list[list[int]] = []
    rows_y: list[list[int]] = []
    buf_x: list[int] = []
    buf_y: list[int] = []

    def _flush():
        nonlocal buf_x, buf_y
        if not buf_x:
            return
        pad = seq_len - len(buf_x)
        if pad:
            buf_x = buf_x + [int(pad_id)] * pad
            buf_y = buf_y + [PADDED_LABEL] * pad
        rows_x.append(buf_x)
        rows_y.append(buf_y)
        buf_x, buf_y = [], []

    for ex in examples:
        ids = list(ex.get("input_ids") or [])
        labs = list(ex.get("labels") or [])
        n = min(len(ids), len(labs), seq_len)
        if n <= 0:
            continue
        ids = ids[:n]
        labs = labs[:n]
        if len(buf_x) + n > seq_len:
            _flush()
        buf_x.extend(ids)
        buf_y.extend(labs)
        if len(buf_x) == seq_len:
            _flush()
    _flush()
    return rows_x, rows_y
