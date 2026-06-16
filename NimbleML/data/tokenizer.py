# tokenizer.py
# Placeholder — a real byte-level BPE tokenizer is planned.
# Char-level helpers live in NimbleML.data.text until then.


class BPETokenizer:
    """Stub API surface for a future BPE tokenizer. All methods raise NotImplementedError."""

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab) if self.vocab else 256

    def _raise(self):
        raise NotImplementedError(
            "BPE tokenizer is not implemented yet (see todo.txt Tier 5.4). "
            "Use char-level helpers in NimbleML.data.text for now."
        )

    def train(self, text, vocab_size, verbose=False, **kwargs):
        self._raise()

    def encode(self, text, verbose=False, **kwargs):
        self._raise()

    def decode(self, ids):
        self._raise()

    def save(self, path):
        self._raise()

    @classmethod
    def load(cls, path):
        cls()._raise()
