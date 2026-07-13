"""Character-level text utilities for language modeling."""

def build_vocab(text):
    """Builds character-to-index and index-to-character mappings.

    Args:
        text (str): Input corpus.

    Returns:
        tuple[dict, list]:
            - char_to_idx: Mapping from character to integer index.
            - idx_to_char: List mapping indices back to characters.
    
    Examples:
        >>> text = "Hello, world!"
        >>> char_to_idx, idx_to_char = build_vocab(text)
    """
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = chars
    return char_to_idx, idx_to_char


def encode(text, char_to_idx):
    """Encodes a string into a list of integer token IDs.

    Args:
        text (str): Input string.
        char_to_idx (dict): Character-to-index mapping.

    Returns:
        list[int]: Encoded token IDs.
    
    Examples:
        >>> text = "Hello, world!"
        >>> char_to_idx = {"H": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "!": 8}
        >>> ids = encode(text, char_to_idx)
    """
    return [char_to_idx[ch] for ch in text]


def decode(ids, idx_to_char):
    """Decodes a list of token IDs back into a string.

    Args:
        ids (list[int]): Token IDs.
        idx_to_char (list): Index-to-character mapping.

    Returns:
        str: Decoded string.
    
    Examples:
        >>> idx_to_char = ["H", "e", "l", "o", " ", "w", "r", "d", "!"]
        >>> ids = [0, 1, 2, 2, 3, 4, 5, 2, 6, 7, 8]
        >>> text = decode(ids, idx_to_char)
    """
    return "".join(idx_to_char[i] for i in ids)


def load_text(path):
    """Loads a text file and builds a character-level vocabulary.

    Args:
        path (str): Path to text file.

    Returns:
        tuple:
            - ids (list[int]): Encoded dataset.
            - char_to_idx (dict): Vocabulary mapping.
            - idx_to_char (list): Reverse vocabulary mapping.
    
    Examples:
        >>> path = "data/text/shakespeare.txt"
        >>> ids, char_to_idx, idx_to_char = load_text(path)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    char_to_idx, idx_to_char = build_vocab(text)
    ids = encode(text, char_to_idx)
    return ids, char_to_idx, idx_to_char


def batch_sequences(ids, batch_size, seq_len=None):
    """Generates batches for next-token prediction training.
    Produces input-target pairs where: input[t] -> target[t+1]

    Supports:
    - Flat token streams
    - Pre-segmented sequences

    Args:
        ids (list[int] or list[list[int]]):
            Token sequence(s).
        batch_size (int):
            Number of sequences per batch.
        seq_len (int, optional):
            Sequence length (required for flat input).

    Yields:
        tuple[Tensor, Tensor]:
            - inputs: shape (batch_size, seq_len)
            - targets: shape (batch_size, seq_len)

    Raises:
        ValueError: If batch_size is less than 1 or seq_len is less than 1.
        ValueError: If ids is a flat list of integers and seq_len is not provided.
    
    Examples:
        >>> ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> batch_size = 2
        >>> seq_len = 3
        >>> inputs, targets = batch_sequences(ids, batch_size, seq_len)
    """
    from NimbleML.data.dataset import SequenceLMDataset, collate_lm_batch, collate_padded_sequences

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if len(ids) == 0:
        return

    if isinstance(ids[0], (list, tuple)):
        dataset = SequenceLMDataset(ids)
        num_batches = len(dataset) // batch_size
        for i in range(num_batches):
            batch = [dataset[i * batch_size + j] for j in range(batch_size)]
            tensors = collate_padded_sequences(batch)
            if tensors is not None:
                yield tensors
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
        rows = []
        for i in range(batch_size):
            chunk = ids[start + i * row_len : start + (i + 1) * row_len]
            rows.append((chunk[:-1], chunk[1:]))
        yield collate_lm_batch(rows)