import os
import random

from NimbleML.activations import Relu, Softmax
from NimbleML.core import eval, parameters, train
from NimbleML.data import load_mnist
from NimbleML.layers import Conv2D, Dense, Flatten, MaxPool2D
from NimbleML.losses import CrossEntropyLoss
from NimbleML.optimizers import Adam
from NimbleML.utils.np_backend import device, np
from NimbleML.utils.tensor import Tensor

DATA_DIR = os.path.join("data", "mnist")
TRAIN_LIMIT = None
TEST_LIMIT = None
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
IMAGE_SIZE = 28
FLAT_FEATURES = 2048


def build_model():
    return [
        Conv2D(1, 32, kernel_size=3),
        Relu(),
        Conv2D(32, 64, kernel_size=3),
        Relu(),
        MaxPool2D(2),
        Conv2D(64, 128, kernel_size=3),
        Relu(),
        Conv2D(128, 128, kernel_size=3),
        Relu(),
        MaxPool2D(2),
        Flatten(),
        Dense(FLAT_FEATURES, 512),
        Relu(),
        Dense(512, 10),
        Softmax(),
    ]


def batch_iter(images, labels, batch_size, shuffle=True):
    indices = list(range(len(images)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_images = [images[i] for i in batch_idx]
        batch_labels = [labels[i] for i in batch_idx]
        flat = [value for image in batch_images for value in image]
        x = Tensor(flat, (len(batch_images), 1, IMAGE_SIZE, IMAGE_SIZE))
        yield x, batch_labels


def forward_logits(model, data):
    for layer in model[:-1]:
        data = layer.forward(data)
    return data


def accuracy(model, images, labels, batch_size):
    correct = 0
    total = 0

    for x, y in batch_iter(images, labels, batch_size, shuffle=False):
        logits = forward_logits(model, x)
        batch, classes = logits.shape
        data = logits.data

        for i in range(batch):
            row = data[i * classes:(i + 1) * classes]
            if int(np.argmax(row)) == y[i]:
                correct += 1
            total += 1

    return correct / max(1, total)


def main():
    (train_images, train_labels), (test_images, test_labels) = load_mnist(DATA_DIR)

    if TRAIN_LIMIT is not None:
        train_images = train_images[:TRAIN_LIMIT]
        train_labels = train_labels[:TRAIN_LIMIT]
    if TEST_LIMIT is not None:
        test_images = test_images[:TEST_LIMIT]
        test_labels = test_labels[:TEST_LIMIT]

    model = build_model()
    loss_fn = CrossEntropyLoss()
    optim = Adam(parameters(model), LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON)

    train(model)
    print(f"Training on {device.upper()}")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        batches = 0
        for x, y in batch_iter(train_images, train_labels, BATCH_SIZE):
            optim.zero_grad()
            logits = forward_logits(model, x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)

        eval(model)
        train_acc = accuracy(model, train_images, train_labels, BATCH_SIZE)
        test_acc = accuracy(model, test_images, test_labels, BATCH_SIZE)

        print(
            f"Epoch {epoch}: "
            f"loss={avg_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )
        train(model)


if __name__ == "__main__":
    main()
