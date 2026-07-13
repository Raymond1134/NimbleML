"""Dataset abstractions for training pipelines."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_int64, np
from NimbleML.utils.tensor import Tensor

# Label value written at padded LM target positions; pass as ignore_index in loss.
PADDED_LABEL = -1


class Dataset(ABC):
    """Abstract dataset: random access to training samples by index."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""

    @abstractmethod
    def __getitem__(self, index: int):
        """Return one sample."""


class InMemoryDataset(Dataset):
    """Dataset backed by a materialized Python sequence."""

    def __init__(self, items: Sequence):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


class TensorDataset(Dataset):
    """Dataset of aligned tensor-like rows (same length along axis 0)."""

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor.")
        length = len(tensors[0])
        for tensor in tensors[1:]:
            if len(tensor) != length:
                raise ValueError("All tensors must have the same length.")
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int):
        return tuple(tensor[index] for tensor in self.tensors)


class TokenLMDataset(Dataset):
    """Next-token windows from a flat token-id stream."""

    def __init__(self, token_ids: Sequence[int], seq_len: int):
        if seq_len < 1:
            raise ValueError("seq_len must be at least 1.")
        self.seq_len = int(seq_len)
        self._ids = as_int64(token_ids).reshape(-1)

    def as_array(self):
        """Return the flat token-id stream as a contiguous int64 ndarray."""
        return self._ids

    def __len__(self) -> int:
        return max(0, int(self._ids.size) - self.seq_len)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        start = index
        window = self._ids[start : start + self.seq_len + 1]
        host = window.get() if hasattr(window, "get") else window
        host = host.tolist()
        return host[:-1], host[1:]

    def batch_from_indices(self, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
        """Build one LM batch by gathering windows at ``indices`` (vectorized)."""
        if not indices:
            raise ValueError("indices must be non-empty.")
        starts = as_int64(indices).reshape(-1)
        offsets = np.arange(self.seq_len + 1, dtype=np.int64)
        windows = self._ids[starts[:, None] + offsets[None, :]]
        batch_size = int(starts.size)
        seq_len = self.seq_len
        inputs = windows[:, :-1].reshape(-1)
        targets = windows[:, 1:].reshape(-1)
        return (
            Tensor.from_int64(inputs, (batch_size, seq_len)),
            Tensor.from_int64(targets, (batch_size, seq_len)),
        )


class SequenceLMDataset(Dataset):
    """One sample per pre-tokenized sequence (variable length)."""

    def __init__(self, sequences: Sequence[Sequence[int]]):
        self.sequences = [list(map(int, seq)) for seq in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


def lm_rows_to_tensors(rows: Sequence[tuple[Sequence[int], Sequence[int]]]) -> tuple[Tensor, Tensor]:
    """Collate LM (input, target) rows into batched int64 tensors."""
    if not rows:
        raise ValueError("rows must be non-empty.")
    batch_size = len(rows)
    seq_len = len(rows[0][0])
    input_data = [token for inputs, _ in rows for token in inputs]
    target_data = [token for _, targets in rows for token in targets]
    return (
        Tensor.from_int64(as_int64(input_data).reshape(batch_size, seq_len), (batch_size, seq_len)),
        Tensor.from_int64(as_int64(target_data).reshape(batch_size, seq_len), (batch_size, seq_len)),
    )


def collate_lm_batch(samples: Sequence[tuple[Sequence[int], Sequence[int]]]) -> tuple[Tensor, Tensor]:
    """Collate ``TokenLMDataset`` samples into one batch."""
    return lm_rows_to_tensors(samples)


def lm_target_mask(real_lens: Sequence[int], seq_len: int) -> np.ndarray:
    """Per-row mask: 1.0 where the LM target is a real token, else 0.0."""
    lens = np.asarray(real_lens, dtype=np.int64)
    positions = np.arange(seq_len, dtype=np.int64)
    return (positions < (lens - 1)[:, None]).astype(np_backend.dtype)


def collate_padded_sequences(sequences: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor] | None:
    """Pad variable-length sequences, then collate for next-token LM training.

    Returns ``(inputs, targets)``. Padded target positions are written as
    :data:`PADDED_LABEL` (``-1``); pass that as ``ignore_index`` in the loss.
    """
    if not sequences:
        return None

    real_lens = [len(seq) for seq in sequences]
    if any(length == 0 for length in real_lens):
        raise ValueError("collate_padded_sequences requires non-empty sequences.")

    max_len = max(real_lens)
    if max_len < 2:
        return None

    seq_len = max_len - 1
    rows = []
    for seq, real_len in zip(sequences, real_lens):
        padded = list(seq)
        if len(padded) < max_len:
            padded = padded + [padded[-1]] * (max_len - len(padded))
        inputs = padded[:-1]
        targets = [
            padded[i + 1] if i < real_len - 1 else PADDED_LABEL
            for i in range(seq_len)
        ]
        rows.append((inputs, targets))

    return lm_rows_to_tensors(rows)
