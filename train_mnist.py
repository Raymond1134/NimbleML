import argparse
import os
import random

from NimbleML.activations import Relu
from NimbleML.core import forward, parameters
from NimbleML.data import load_mnist
from NimbleML.layers import Dense
from NimbleML.losses import CrossEntropyLoss
from NimbleML.optimizers import NAG, SGD, SGDM
from NimbleML.utils.tensor import Tensor

OPTIMIZERS = {
    "sgd": SGD,
    "sgdm": SGDM,
    "nag": NAG,
}

def batch_iter(images, labels, batch_size, shuffle=True):
    indices = list(range(len(images)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_images = [images[i] for i in batch_idx]
        batch_labels = [labels[i] for i in batch_idx]
        flat = [value for image in batch_images for value in image]
        x = Tensor(flat, (len(batch_images), len(batch_images[0])))
        yield x, batch_labels

def accuracy(model, images, labels, batch_size):
    correct = 0
    total = 0

    for x, y in batch_iter(images, labels, batch_size, shuffle=False):
        logits = forward(model, x)
        batch, classes = logits.shape
        data = logits.data

        for i in range(batch):
            row = data[i * classes:(i + 1) * classes]
            pred = max(range(classes), key=lambda j: row[j])
            if pred == y[i]:
                correct += 1
            total += 1

    return correct / max(1, total)

def main():
    parser = argparse.ArgumentParser(description="Train a tiny MNIST MLP.")
    parser.add_argument("--data-dir", default=os.path.join("data", "mnist"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="nag")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--test-limit", type=int, default=250)
    args = parser.parse_args()

    (train_images, train_labels), (test_images, test_labels) = load_mnist(args.data_dir)

    if args.train_limit:
        train_images = train_images[:args.train_limit]
        train_labels = train_labels[:args.train_limit]
    if args.test_limit:
        test_images = test_images[:args.test_limit]
        test_labels = test_labels[:args.test_limit]

    model = [
        Dense(784, 256),
        Relu(),
        Dense(256, 64),
        Relu(),
        Dense(64, 10),
    ]

    loss_fn = CrossEntropyLoss()
    optim_cls = OPTIMIZERS[args.optimizer]
    if args.optimizer == "sgd":
        optim = optim_cls(parameters(model), args.lr)
    else:
        optim = optim_cls(parameters(model), args.lr, args.momentum)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batches = 0
        for x, y in batch_iter(train_images, train_labels, args.batch_size):
            optim.zero_grad()
            logits = forward(model, x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            batches += 1

        acc = accuracy(model, test_images, test_labels, args.batch_size)
        avg_loss = total_loss / max(1, batches)

        train_acc = accuracy(
            model,
            train_images,
            train_labels,
            args.batch_size
        )

        test_acc = accuracy(
            model,
            test_images,
            test_labels,
            args.batch_size
        )

        print(
            f"Epoch {epoch}: "
            f"loss={avg_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
