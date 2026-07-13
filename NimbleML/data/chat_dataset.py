"""Chat / instruction datasets with assistant-only loss masks."""
from __future__ import annotations

from NimbleML.data.dataset import Dataset, PADDED_LABEL
from NimbleML.utils.np_backend import np


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
        return tokenizer.encode(text)
    raise TypeError("tokenizer must provide encode()")


def build_sft_example(messages: list[dict], tokenizer, *, max_seq_len: int):
    """Build input_ids + labels where only assistant token spans contribute to loss."""
    ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        role = m["role"]
        chunk = f"<|{role}|>\n{m['content']}<|end|>\n"
        tok = list(tokenizer.encode(chunk))
        ids.extend(tok)
        if role == "assistant":
            labels.extend(tok)
        else:
            labels.extend([PADDED_LABEL] * len(tok))
    ids = ids[:max_seq_len]
    labels = labels[:max_seq_len]
    # next-token shift for LM: labels[t] predicts ids[t] from prefix — keep aligned
    # for compute_loss which expects per-position class labels matching logits.
    return {"input_ids": ids, "labels": labels}
