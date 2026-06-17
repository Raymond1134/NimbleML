"""Gradcheck tests (float64 only — not used in GPU training)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NimbleML.layers.dense import Dense
from NimbleML.utils.gradcheck import gradcheck
from NimbleML.utils.np_backend import np, set_dtype
from NimbleML.utils.tensor import Tensor

set_dtype("float64")


def test_dense_gradcheck():
    layer = Dense(4, 3)
    x = Tensor(np.linspace(0.1, 0.8, 8, dtype=np.float64), (2, 4), requires_grad=True)
    gradcheck(
        lambda: layer.forward(x).sum(),
        [x, layer.weights, layer.biases],
        tol=1e-3,
    )


def main():
    test_dense_gradcheck()
    print("Gradcheck tests passed.")


if __name__ == "__main__":
    main()
