# text.py
# Text data processing utilities


def build_vocab(text):
    """
    Build char <-> index maps from a string.

    Returns:
        char_to_idx: dict mapping each unique character to an int
        idx_to_char: list where index i gives the character for i
    """
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
    """Read a text file, build vocab, and return encoded ids plus vocab maps."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    char_to_idx, idx_to_char = build_vocab(text)
    ids = encode(text, char_to_idx)
    return ids, char_to_idx, idx_to_char
