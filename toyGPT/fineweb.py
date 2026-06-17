"""FineWeb-Edu data preparation: stream -> fast BPE -> memmapped uint16 bins.

Pipeline (all streaming / bounded-memory):

1. ``ensure_tokenizer`` trains (or loads) a byte-level BPE on a text sample.
2. ``load_or_build_corpus`` streams documents, encodes them in parallel, and
   appends token ids to ``train.bin`` / ``val.bin`` as ``uint16`` (vocab < 65536
   so 2 bytes/token). The corpus never lives in RAM as a Python list, and is
   memmapped at train time so only sampled batches touch memory / the GPU.

Everything is keyed by a content hash (dataset + budgets + tokenizer bytes), so
re-runs and resumes skip both download and re-encoding.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np  # host NumPy: the corpus stays on the CPU

from toyGPT.fast_tokenizer import FastBPETokenizer

DATA_FORMAT_VERSION = 1
TOKEN_DTYPE = np.uint16  # vocab_size must be <= 65536


def _doc_stream(cfg) -> Iterator[str]:
    """Yield non-empty document texts from the configured HF dataset (streaming)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_repo, name=cfg.hf_subset, split="train", streaming=True)
    for ex in ds:
        text = ex.get("text")
        if text:
            yield text


def _batched(iterable: Iterable, n: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def _sample_text_iter(cfg) -> Iterator[str]:
    """Stream documents until ~``tokenizer_sample_mb`` MB of text is yielded."""
    budget = int(cfg.tokenizer_sample_mb) * 1024 * 1024
    used = 0
    for text in _doc_stream(cfg):
        yield text
        used += len(text)
        if used >= budget:
            return


def ensure_tokenizer(cfg, *, resume_dir: Optional[Path] = None, verbose: bool = True):
    """Return ``(tokenizer, tokenizer_path)``, training a new BPE only if needed.

    On resume we mirror the checkpoint's tokenizer to ``cfg.tokenizer_path`` so
    the corpus cache key stays stable across sessions.
    """
    tok_path = Path(cfg.tokenizer_path)

    if resume_dir is not None:
        ckpt_tok = Path(resume_dir) / "tokenizer.json"
        if ckpt_tok.is_file():
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            if not tok_path.is_file() or tok_path.read_bytes() != ckpt_tok.read_bytes():
                shutil.copyfile(ckpt_tok, tok_path)
            if verbose:
                print(f"[data] loading tokenizer from checkpoint {ckpt_tok}")
            return FastBPETokenizer.load(tok_path), tok_path

    if tok_path.is_file():
        if verbose:
            print(f"[data] loading tokenizer from {tok_path}")
        return FastBPETokenizer.load(tok_path), tok_path

    if int(cfg.vocab_size) > 65536:
        raise ValueError("vocab_size must be <= 65536 to store token ids as uint16.")

    if verbose:
        print(
            f"[data] training BPE | vocab_size={cfg.vocab_size} "
            f"sample~{cfg.tokenizer_sample_mb}MB dataset={cfg.dataset}/{cfg.hf_subset}"
        )
    t0 = time.perf_counter()
    tok = FastBPETokenizer.train(
        _sample_text_iter(cfg),
        vocab_size=int(cfg.vocab_size),
        save_path=tok_path,
    )
    if verbose:
        print(
            f"[data] tokenizer trained | vocab={tok.vocab_size} "
            f"elapsed={time.perf_counter() - t0:.1f}s -> {tok_path}"
        )
    return tok, tok_path


def corpus_cache_key(cfg, tokenizer: FastBPETokenizer) -> str:
    """Stable cache key from dataset + token budgets + tokenizer content.

    Keyed on the tokenizer's canonical vocab fingerprint (not file bytes) so a
    resume reuses the cached bins even if ``tokenizer.json`` was re-serialized.
    """
    h = hashlib.sha256()
    head = (
        f"v{DATA_FORMAT_VERSION}|{cfg.dataset}|{cfg.hf_subset}|"
        f"vocab={cfg.vocab_size}|train={cfg.train_tokens}|val={cfg.val_tokens}|"
        f"tok={tokenizer.fingerprint()}"
    )
    h.update(head.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_dir(cfg, key: str) -> Path:
    return Path(cfg.cache_dir) / "encoded" / cfg.dataset / key


def _read_meta(meta_path: Path) -> Optional[dict]:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_chunk(handle, buf: list[int]) -> None:
    if buf:
        handle.write(np.asarray(buf, dtype=TOKEN_DTYPE).tobytes())


def _build_bins(
    cfg,
    tokenizer: FastBPETokenizer,
    train_path: Path,
    val_path: Path,
    *,
    verbose: bool = True,
    doc_batch: int = 512,
    flush_tokens: int = 8_000_000,
) -> tuple[int, int]:
    """Stream + encode documents into uint16 bins; return (train_tokens, val_tokens)."""
    eot = tokenizer.eot_id
    val_target = int(cfg.val_tokens)
    train_target = int(cfg.train_tokens)

    tmp_train = train_path.with_name(train_path.name + ".tmp")
    tmp_val = val_path.with_name(val_path.name + ".tmp")

    n_val = 0
    n_train = 0
    val_buf: list[int] = []
    train_buf: list[int] = []
    t0 = time.perf_counter()
    last_log = t0
    done = False

    if verbose:
        print(
            f"[data] encoding corpus -> train={train_target:,} val={val_target:,} tokens "
            f"(dtype=uint16, doc_batch={doc_batch})"
        )

    with open(tmp_val, "wb") as fval, open(tmp_train, "wb") as ftrain:
        for batch in _batched(_doc_stream(cfg), doc_batch):
            for ids in tokenizer.encode_batch(batch):
                ids.append(eot)
                if n_val < val_target:
                    val_buf.extend(ids)
                    n_val += len(ids)
                    if len(val_buf) >= flush_tokens:
                        _write_chunk(fval, val_buf)
                        val_buf.clear()
                else:
                    train_buf.extend(ids)
                    n_train += len(ids)
                    if len(train_buf) >= flush_tokens:
                        _write_chunk(ftrain, train_buf)
                        train_buf.clear()
                    if n_train >= train_target:
                        done = True
                        break
            if verbose and (time.perf_counter() - last_log) > 10.0:
                done_tok = n_val + n_train
                rate = done_tok / max(1e-9, time.perf_counter() - t0)
                eta = (val_target + train_target - done_tok) / max(1.0, rate)
                print(
                    f"[data]   progress | val={n_val:,}/{val_target:,} "
                    f"train={n_train:,}/{train_target:,} | {rate/1e6:.2f} Mtok/s "
                    f"eta={eta/60:.1f} min"
                )
                last_log = time.perf_counter()
            if done:
                break

        _write_chunk(fval, val_buf)
        _write_chunk(ftrain, train_buf)

    if not done and verbose:
        print(
            f"[data] warning: dataset exhausted before train budget "
            f"({n_train:,}/{train_target:,} tokens). Using what was collected."
        )

    os.replace(tmp_val, val_path)
    os.replace(tmp_train, train_path)
    if verbose:
        print(
            f"[data] encode done | train_tokens={n_train:,} val_tokens={n_val:,} "
            f"elapsed={time.perf_counter() - t0:.1f}s"
        )
    return n_train, n_val


def load_or_build_corpus(
    cfg,
    tokenizer: FastBPETokenizer,
    tokenizer_path: Path,
    *,
    verbose: bool = True,
) -> tuple[Path, Path, dict]:
    """Return ``(train_path, val_path, meta)``, building the bins only if missing."""
    key = corpus_cache_key(cfg, tokenizer)
    cache_dir = _cache_dir(cfg, key)
    train_path = cache_dir / "train.bin"
    val_path = cache_dir / "val.bin"
    meta_path = cache_dir / "meta.json"

    meta = _read_meta(meta_path)
    if (
        meta
        and meta.get("key") == key
        and meta.get("version") == DATA_FORMAT_VERSION
        and train_path.is_file()
        and val_path.is_file()
    ):
        if verbose:
            print(
                f"[data] using cached corpus | train_tokens={meta['train_tokens']:,} "
                f"val_tokens={meta['val_tokens']:,} cache={cache_dir}"
            )
        return train_path, val_path, meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_val = _build_bins(cfg, tokenizer, train_path, val_path, verbose=verbose)
    meta = {
        "version": DATA_FORMAT_VERSION,
        "key": key,
        "dataset": cfg.dataset,
        "hf_subset": cfg.hf_subset,
        "dtype": "uint16",
        "vocab_size": int(tokenizer.vocab_size),
        "eot_id": int(tokenizer.eot_id),
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "tokenizer_path": str(tokenizer_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"[data] saved corpus cache -> {cache_dir}")
    return train_path, val_path, meta


def load_token_bin(path: Path) -> np.memmap:
    """Memmap a uint16 token bin as a read-only host array (near-zero RAM)."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def prepare_corpus(cfg, *, resume_dir: Optional[Path] = None, verbose: bool = True):
    """End-to-end prep: ``(tokenizer, tokenizer_path, train_path, val_path, meta)``."""
    tokenizer, tok_path = ensure_tokenizer(cfg, resume_dir=resume_dir, verbose=verbose)
    train_path, val_path, meta = load_or_build_corpus(
        cfg, tokenizer, tok_path, verbose=verbose
    )
    return tokenizer, tok_path, train_path, val_path, meta
