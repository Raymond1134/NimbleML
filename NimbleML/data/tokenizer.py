# tokenizer.py
# Byte-level BPE tokenizer (GPT-2 style) with regex pre-tokenization, from scratch.
import json
import re
import time

# GPT-2-style pre-tokenization: split contractions, words (optional leading
# space), number runs, punctuation runs, and whitespace. Pre-tokenizing keeps
# merges from crossing word boundaries and lets encode() cache per-chunk work.
_SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def _get_stats(ids, counts=None):
    """Count occurrences of each adjacent pair in a sequence of ids."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    n = len(ids)
    first, second = pair
    while i < n:
        if i < n - 1 and ids[i] == first and ids[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level Byte-Pair Encoding tokenizer with GPT-2-style pre-tokenization.

    Works on raw UTF-8 bytes, so any string round-trips with no unknown token.
    The base vocabulary is the 256 byte values; training learns merges on top.
    """

    def __init__(self):
        # merges: (id_a, id_b) -> new_id (insertion order == merge rank)
        self.merges = {}
        # vocab: id -> bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        # cache: chunk string -> encoded ids (filled lazily during encode)
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False, log_every=25):
        """Learn BPE merges from `text` until the vocab reaches `vocab_size`."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (byte-level base vocab).")

        num_merges = vocab_size - 256
        if verbose:
            print(f"  Pre-tokenizing {len(text):,} chars...")
        chunks = [list(piece.encode("utf-8")) for piece in _SPLIT_PATTERN.findall(text)]
        if verbose:
            print(f"  {len(chunks):,} chunks | learning {num_merges:,} merges (log every {log_every})...")

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        train_start = time.time()
        last_log = train_start

        for i in range(num_merges):
            stats = {}
            for chunk in chunks:
                _get_stats(chunk, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            chunks = [_merge(chunk, pair, new_id) for chunk in chunks]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                now = time.time()
                step = i + 1
                if step == 1 or step % log_every == 0 or step == num_merges or (now - last_log) >= 30:
                    elapsed = now - train_start
                    rate = step / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  merge {step:,}/{num_merges:,} | pair {pair} -> {new_id} "
                        f"| count {stats[pair]:,} | {elapsed:.1f}s ({rate:.1f} merge/s)"
                    )
                    last_log = now

        self.merges = merges
        self.vocab = vocab
        self._cache = {}
        if verbose:
            print(f"  BPE training done: vocab={len(vocab):,} in {time.time() - train_start:.1f}s")
        return self

    def _encode_chunk(self, piece):
        """Encode a single pre-token chunk (string), memoized."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # Pick the mergeable pair with the lowest merge rank (earliest learned).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])

        self._cache[piece] = ids
        return ids

    def encode(self, text, verbose=False, log_every_chunks=10_000, log_every_seconds=30):
        """Encode a string into a list of token ids."""
        pieces = _SPLIT_PATTERN.findall(text)
        total_pieces = len(pieces)
        ids = []
        encode_start = time.time()
        last_log = encode_start

        if verbose:
            print(f"  Encoding {len(text):,} chars in {total_pieces:,} chunks...")

        for i, piece in enumerate(pieces):
            ids.extend(self._encode_chunk(piece))
            if not verbose:
                continue
            now = time.time()
            step = i + 1
            if (
                step == 1
                or step == total_pieces
                or step % log_every_chunks == 0
                or (now - last_log) >= log_every_seconds
            ):
                elapsed = now - encode_start
                pct = 100.0 * step / total_pieces if total_pieces else 100.0
                rate = step / elapsed if elapsed > 0 else 0.0
                print(
                    f"  encode {step:,}/{total_pieces:,} chunks ({pct:.1f}%) "
                    f"| {len(ids):,} tokens | {elapsed:.1f}s ({rate:.0f} chunk/s)"
                )
                last_log = now

        if verbose:
            print(f"  Encoding done: {len(ids):,} tokens in {time.time() - encode_start:.1f}s")
        return ids

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        tokens = b"".join(self.vocab[int(i)] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, path):
        """Persist merges to a JSON file."""
        payload = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    @classmethod
    def load(cls, path):
        """Load a tokenizer previously written by `save`."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls()
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in payload["merges"]:
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]
        tok.merges = merges
        tok.vocab = vocab
        return tok
