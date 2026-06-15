# text.py
# Text utilities for language modeling (char-level and BPE token-level)
import os
from NimbleML.data.tokenizer import BPETokenizer
from NimbleML.utils.tensor import Tensor


def build_vocab(text):
    """Build char <-> index maps from a string."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = chars
    return char_to_idx, idx_to_char


def encode(text, char_to_idx):
    """Convert a string to a list of character indices."""
    return [char_to_idx[ch] for ch in text]


def decode(ids, idx_to_char):
    """Convert a list of character indices back to a string."""
    return "".join(idx_to_char[i] for i in ids)


def load_text(path):
    """Read a text file, build char vocab, return encoded ids plus vocab maps."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    char_to_idx, idx_to_char = build_vocab(text)
    ids = encode(text, char_to_idx)
    return ids, char_to_idx, idx_to_char


def _rows_to_tensors(rows):
    """Split full token rows into input/target pairs and wrap as Tensors."""
    inputs = [row[:-1] for row in rows]
    targets = [row[1:] for row in rows]
    batch_size = len(rows)
    seq_len = len(inputs[0])
    input_data = [token for row in inputs for token in row]
    target_data = [token for row in targets for token in row]
    return (
        Tensor(input_data, (batch_size, seq_len)),
        Tensor(target_data, (batch_size, seq_len)),
    )


def batch_sequences(ids, batch_size, seq_len=None):
    """Yield (inputs, targets) for next-token prediction."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if ids and isinstance(ids[0], list):
        num_batches = len(ids) // batch_size
        for i in range(num_batches):
            batch = ids[i * batch_size:(i + 1) * batch_size]
            max_len = max(len(seq) for seq in batch)
            if max_len < 2:
                continue
            rows = []
            for seq in batch:
                if len(seq) < max_len:
                    seq = seq + [seq[-1]] * (max_len - len(seq))
                rows.append(seq)
            yield _rows_to_tensors(rows)
        return

    if seq_len is None:
        raise ValueError("seq_len is required when ids is a flat list of integers.")
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1.")

    row_len = seq_len + 1
    batch_stride = batch_size * row_len
    num_batches = len(ids) // batch_stride

    for b in range(num_batches):
        start = b * batch_stride
        rows = [
            ids[start + i * row_len:start + (i + 1) * row_len]
            for i in range(batch_size)
        ]
        yield _rows_to_tensors(rows)


def _encode_corpus(text, tokenizer):
    """Encode a full corpus. The tokenizer caches per pre-token chunk, so repeated
    words across the corpus are encoded only once."""
    return tokenizer.encode(text)


def load_text_bpe(path, tokenizer_path=None, vocab_size=1024, max_train_chars=1_000_000, verbose=True):
    """Load a corpus as BPE token ids, training (and caching) the tokenizer if needed.

    Returns (ids, tokenizer). If `tokenizer_path` exists it is loaded; otherwise a
    new tokenizer is trained on the corpus and saved to `tokenizer_path` (if given).

    The naive BPE trainer is O(merges * corpus) in pure Python, so merges are learned
    from at most `max_train_chars` characters of the corpus (the full corpus is still
    encoded). Set `max_train_chars=0` to train on everything.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if tokenizer_path is not None and os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
        if verbose:
            print(f"Loaded tokenizer from {tokenizer_path} (vocab={tokenizer.vocab_size}).")
    else:
        train_text = text if max_train_chars <= 0 else text[:max_train_chars]
        if verbose:
            print(
                f"Training BPE tokenizer (target vocab={vocab_size}) "
                f"on {len(train_text):,} chars..."
            )
        tokenizer = BPETokenizer().train(train_text, vocab_size, verbose=verbose)
        if tokenizer_path is not None:
            tokenizer.save(tokenizer_path)
            if verbose:
                print(f"Saved tokenizer to {tokenizer_path}.")

    if verbose:
        print("Encoding corpus...")
    ids = _encode_corpus(text, tokenizer)
    return ids, tokenizer
