"""Placeholder — a real byte-level BPE tokenizer is planned."""
# Char-level helpers live in NimbleML.data.text until then.


class BPETokenizer:
    """Stub API surface for a future BPE tokenizer. All methods raise NotImplementedError."""

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self._cache = {}

    @property
    def vocab_size(self):
        """Public function vocab_size."""
        return len(self.vocab) if self.vocab else 256

    def _raise(self):
        raise NotImplementedError(
            "BPE tokenizer is not implemented yet (see todo.txt Tier 5.4). "
            "Use char-level helpers in NimbleML.data.text for now."
        )

    def train(self, text, vocab_size, verbose=False, **kwargs):
        """Public function train."""
        self._raise()

    def encode(self, text, verbose=False, **kwargs):
        """Public function encode."""
        self._raise()

    def decode(self, ids):
        """Public function decode."""
        self._raise()

    def save(self, path):
        """Public function save."""
        self._raise()

    @classmethod
    def load(cls, path):
        """Public function load."""
        cls()._raise()
