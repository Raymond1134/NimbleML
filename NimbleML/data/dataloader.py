"""DataLoader: batched iteration over :class:`Dataset` samples."""
from __future__ import annotations
import queue
import random
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from .dataset import (
    Dataset,
    SequenceLMDataset,
    TokenLMDataset,
    collate_lm_batch,
    collate_padded_sequences,
)

_SENTINEL = object()


def default_collate(samples: Sequence[Any]) -> Any:
    """Collate a list of dataset samples into one batch when possible."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    first = samples[0]
    if isinstance(first, tuple) and len(first) == 2:
        a, b = first
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            if all(
                isinstance(x, list)
                and isinstance(y, list)
                and len(x) == len(a)
                and len(y) == len(b)
                for x, y in samples
            ):
                return collate_lm_batch(samples)

    if isinstance(first, list) and first and isinstance(first[0], int):
        if all(isinstance(seq, list) and seq and isinstance(seq[0], int) for seq in samples):
            result = collate_padded_sequences(samples)
            if result is not None:
                return result

    return list(samples)


class DataLoader:
    """Iterate a :class:`Dataset` in collated batches.

    Args:
        dataset: Source of training samples.
        batch_size: Samples per batch.
        shuffle: Reshuffle indices at the start of each iteration.
        drop_last: Omit the final partial batch when the dataset size is not
            divisible by ``batch_size``.
        collate_fn: Merge a list of samples into one batch. When ``None``,
            picks a collate function based on ``dataset`` type.
        num_workers: Worker processes for sample loading. Only ``0`` is supported.
        prefetch_factor: Batches to prepare ahead in a background thread when
            ``num_workers`` is 0. Set to 0 to disable prefetch.
        seed: Optional RNG seed used when ``shuffle`` is True.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        seed: int | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if num_workers != 0:
            raise NotImplementedError("num_workers > 0 is not supported yet.")
        if prefetch_factor < 0:
            raise ValueError("prefetch_factor must be >= 0.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn if collate_fn is not None else self._collate_for_dataset(dataset)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.seed = seed

    @staticmethod
    def _collate_for_dataset(dataset: Dataset) -> Callable[[Sequence[Any]], Any]:
        if isinstance(dataset, TokenLMDataset):
            return collate_lm_batch
        if isinstance(dataset, SequenceLMDataset):
            def _collate(samples: Sequence[Any]) -> Any:
                result = collate_padded_sequences(samples)
                if result is None:
                    raise ValueError("Sequence batch is too short for LM collation.")
                return result

            return _collate
        return default_collate

    def __len__(self) -> int:
        n = len(self.dataset)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _batch_index_lists(self) -> list[list[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        if self.drop_last:
            limit = len(indices) // self.batch_size * self.batch_size
            indices = indices[:limit]

        batches = []
        for start in range(0, len(indices), self.batch_size):
            batches.append(indices[start : start + self.batch_size])
        return batches

    def _iter_batches(self) -> Iterator[Any]:
        # TokenLMDataset always gathers windows with a single ndarray index —
        # that is the only batching path (no separate slow __getitem__ loop).
        if isinstance(self.dataset, TokenLMDataset) and self.collate_fn is collate_lm_batch:
            for batch_indices in self._batch_index_lists():
                yield self.dataset.batch_from_indices(batch_indices)
            return

        for batch_indices in self._batch_index_lists():
            samples = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(samples)

    def __iter__(self) -> Iterator[Any]:
        if self.prefetch_factor == 0:
            yield from self._iter_batches()
            return

        batch_queue: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_factor)
        error_box: list[BaseException] = []

        def producer() -> None:
            try:
                for batch in self._iter_batches():
                    batch_queue.put(batch)
            except BaseException as exc:
                error_box.append(exc)
            finally:
                batch_queue.put(_SENTINEL)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = batch_queue.get()
            if item is _SENTINEL:
                break
            yield item

        thread.join()
        if error_box:
            raise error_box[0]
