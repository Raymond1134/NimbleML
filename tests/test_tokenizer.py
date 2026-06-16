"""Tokenizer unit tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NimbleML.data.tokenizer import BYTE_VOCAB_SIZE, BPETokenizer


def test_bpe_roundtrip_toy_corpus():
    text = "aaabdaaabac" * 50
    tok = BPETokenizer()
    tok.train(text, vocab_size=300, verbose=False)
    encoded = tok.encode(text)
    assert tok.decode(encoded) == text


def test_bpe_empty_string():
    tok = BPETokenizer()
    tok.train("abc", vocab_size=260)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_bpe_unicode_roundtrip():
    text = "hello 世界 🚀"
    tok = BPETokenizer()
    tok.train(text * 100, vocab_size=400)
    assert tok.decode(tok.encode(text)) == text


def test_bpe_save_load_roundtrip(tmp_path=None):
    if tmp_path is None:
        tmp_path = Path("tests_tmp_tokenizer")
        tmp_path.mkdir(exist_ok=True)
        path = tmp_path / "tokenizer.json"
    else:
        path = tmp_path / "tokenizer.json"

    text = "the quick brown fox jumps over the lazy dog" * 20
    tok = BPETokenizer()
    tok.train(text, vocab_size=350)
    tok.save(path)

    loaded = BPETokenizer.load(path)
    sample = "the quick fox"
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(tok.encode(sample)) == sample


def test_bpe_vocab_size_floor():
    tok = BPETokenizer()
    tok.train("abc", vocab_size=BYTE_VOCAB_SIZE)
    assert tok.vocab_size == BYTE_VOCAB_SIZE


def test_bpe_encode_batch():
    tok = BPETokenizer()
    tok.train("aaabdaaabac" * 30, vocab_size=280)
    texts = ["aaa", "bac"]
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]


def main():
    test_bpe_roundtrip_toy_corpus()
    test_bpe_empty_string()
    test_bpe_unicode_roundtrip()
    test_bpe_save_load_roundtrip()
    test_bpe_vocab_size_floor()
    test_bpe_encode_batch()
    print("Tokenizer tests passed.")


if __name__ == "__main__":
    main()
