"""WikiText download and char-level batch sampling for toy GPT."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from NimbleML.data.text import build_vocab, decode, encode
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor

WIKITEXT_URLS = {
    "wikitext-2": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1.zip",
    "wikitext-103": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1.zip",
}

# Reliable per-split mirrors when zip hosting is unavailable.
WIKITEXT_SPLIT_URLS = {
    "wikitext-2": {
        "wiki.train.raw": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "wiki.valid.raw": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "wiki.test.raw": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    },
}


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "NimbleML-toyGPT"})
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as out:
        out.write(response.read())


def _download_split_files(dataset: str, extract_dir: Path) -> Path:
    splits = WIKITEXT_SPLIT_URLS.get(dataset)
    if splits is None:
        raise RuntimeError(f"No split download URLs configured for {dataset!r}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in splits.items():
        dest = extract_dir / filename
        if not dest.is_file():
            _download_file(url, dest)
    return extract_dir


def _local_wikitext_dirs(dataset: str, cache_dir: Path, data_dir: Path) -> list[Path]:
    folder = f"{dataset}-raw"
    repo_root = Path(__file__).resolve().parents[1]
    return [
        cache_dir / folder,
        data_dir / folder,
        repo_root / "NimbleML" / "data" / folder,
    ]


def download_wikitext(dataset: str, cache_dir: Path, data_dir: Path) -> Path:
    """Download and extract WikiText raw splits; return path to extracted folder."""
    if dataset not in WIKITEXT_URLS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(WIKITEXT_URLS)}")

    folder_name = f"{dataset}-raw"
    for candidate in _local_wikitext_dirs(dataset, cache_dir, data_dir):
        train_file = candidate / "wiki.train.raw"
        if train_file.is_file():
            return candidate

    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / folder_name
    train_file = extract_dir / "wiki.train.raw"

    if dataset in WIKITEXT_SPLIT_URLS:
        return _download_split_files(dataset, extract_dir)

    url = WIKITEXT_URLS[dataset]
    zip_path = cache_dir / f"{folder_name}.zip"
    if not zip_path.is_file():
        print(f"Downloading {dataset} from {url} ...")
        _download_file(url, zip_path)

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    if not train_file.is_file():
        raise FileNotFoundError(f"Expected {train_file} after extracting {zip_path}")
    return extract_dir


def load_wikitext_splits(dataset: str, cache_dir: Path, data_dir: Path) -> tuple[str, str]:
    """Return (train_text, val_text) for a WikiText dataset."""
    root = download_wikitext(dataset, cache_dir, data_dir)
    train_text = (root / "wiki.train.raw").read_text(encoding="utf-8")
    val_text = (root / "wiki.valid.raw").read_text(encoding="utf-8")
    return train_text, val_text


def encode_with_vocab(text: str, char_to_idx: dict) -> list[int]:
    """Encode text with a fixed char vocabulary (skips unknown characters)."""
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def prepare_char_corpus(text: str) -> tuple[list[int], dict, list]:
    """Build char vocab and encode a text string."""
    char_to_idx, idx_to_char = build_vocab(text)
    ids = encode(text, char_to_idx)
    return ids, char_to_idx, idx_to_char


def random_batch(
    ids: list[int],
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample a random training batch from a flat token array."""
    row_len = seq_len + 1
    if len(ids) < row_len + 1:
        raise ValueError("Corpus too short for the requested seq_len.")

    max_start = len(ids) - row_len
    starts = [int(x) for x in rng.integers(0, max_start + 1, size=batch_size)]
    rows = [ids[start : start + row_len] for start in starts]
    flat_in = [token for row in rows for token in row[:-1]]
    flat_tgt = [token for row in rows for token in row[1:]]
    return (
        Tensor(flat_in, (batch_size, seq_len)),
        Tensor(flat_tgt, (batch_size, seq_len)),
    )
