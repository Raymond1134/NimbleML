# NimbleML
A lightweight Python machine learning library for building and training simple models. NimbleML provides clean implementations of core ML concepts such as linear/logistic regression, basic neural networks, loss functions, gradient-based optimization, and more.

## Quick start (MNIST)
Run the minimal MNIST training script (downloads data to data/mnist):

```bash
python -m pip install numpy
python train_mnist.py --epochs 1 --train-limit 1000 --test-limit 200
```
