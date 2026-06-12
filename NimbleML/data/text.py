# text.py
# Text data processing utilities (character and word tokenization)
import re
from collections import Counter

from NimbleML.utils.tensor import Tensor

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

_WORD_PATTERN = re.compile(r"[a-z0-9']+|[.,!?;:]")


def tokenize_words(text):
    """Split text into lowercase word/punctuation tokens."""
    return _WORD_PATTERN.findall(text.lower())


# --- Character-level (legacy / tests) ---


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


# --- Word-level ---


def build_word_vocab(text, max_vocab=8000):
    """Build word vocab from most frequent tokens. Index 0 = UNK, 1 = PAD."""
    counts = Counter(tokenize_words(text))
    vocab = [UNK_TOKEN, PAD_TOKEN]
    for word, _ in counts.most_common(max(2, max_vocab) - len(vocab)):
        if word not in vocab:
            vocab.append(word)
    word_to_idx = {word: index for index, word in enumerate(vocab)}
    return word_to_idx, vocab


def encode_words(text, word_to_idx):
    """Convert text to word indices; unknown words map to UNK."""
    unk_id = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(word, unk_id) for word in tokenize_words(text)]


def decode_words(ids, idx_to_word):
    """Join word indices back into readable text."""
    return " ".join(idx_to_word[i] for i in ids)


def load_word_text(path, max_vocab=8000):
    """Read file, build word vocab, return encoded ids plus vocab maps."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    word_to_idx, idx_to_word = build_word_vocab(text, max_vocab=max_vocab)
    ids = encode_words(text, word_to_idx)
    return ids, word_to_idx, idx_to_word


def encode_prompt(prompt, word_to_idx):
    """Encode a user prompt that may contain unknown words as UNK."""
    return encode_words(prompt, word_to_idx)


# --- Batching (token-agnostic) ---


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
    """Yield (inputs, targets) for next-token prediction.

    Flat ids (from load_text / load_word_text): pass seq_len.
    List of sequences: batches are padded to the longest sequence in the batch.
    """
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
