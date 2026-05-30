import gzip
import os
import struct
import urllib.request


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    paths = {}

    for key, filename in FILES.items():
        path = os.path.join(dest_dir, filename)
        if not os.path.exists(path):
            url = BASE_URL + filename
            urllib.request.urlretrieve(url, path)
        paths[key] = path

    return paths


def _read_idx_images(path, normalize=True):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid IDX image file.")
        data = f.read()

    image_size = rows * cols
    images = []
    for i in range(count):
        start = i * image_size
        raw = data[start:start + image_size]
        if normalize:
            images.append([b / 255.0 for b in raw])
        else:
            images.append(list(raw))

    return images


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid IDX label file.")
        data = f.read()

    return list(data[:count])


def load_mnist(dest_dir, normalize=True):
    paths = download_mnist(dest_dir)

    train_images = _read_idx_images(paths["train_images"], normalize=normalize)
    train_labels = _read_idx_labels(paths["train_labels"])
    test_images = _read_idx_images(paths["test_images"], normalize=normalize)
    test_labels = _read_idx_labels(paths["test_labels"])

    return (train_images, train_labels), (test_images, test_labels)
