from NimbleML.layers.conv2D import Conv2D, _im2col
from NimbleML.layers.flatten import Flatten
from NimbleML.layers import MaxPool2D
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def _assert_list_close(label, actual, expected, tol=1e-6):
	if len(actual) != len(expected):
		raise AssertionError(f"{label}: length {len(actual)} != {len(expected)}")
	for i, (a, e) in enumerate(zip(actual, expected)):
		if abs(a - e) > tol:
			raise AssertionError(f"{label}[{i}]: {a} != {e}")


def test_forward_broadcasting():
	a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
	b = Tensor([10, 20, 30], (3,))
	out = a + b
	_assert_list_close("(2,3)+(3,) data", out.data, [11, 22, 33, 14, 25, 36])

	a = Tensor([1, 2], (2, 1))
	b = Tensor([10, 20, 30], (1, 3))
	out = a + b
	_assert_list_close("(2,1)+(1,3) data", out.data, [11, 21, 31, 12, 22, 32])

	s = Tensor([2.0], ())
	m = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
	out = s + m
	_assert_list_close("scalar+(2,3) data", out.data, [3, 4, 5, 6, 7, 8])

	out = m + s
	_assert_list_close("(2,3)+scalar data", out.data, [3, 4, 5, 6, 7, 8])


def test_backward_broadcasting():
	a = Tensor([1, 2], (2, 1), requires_grad=True)
	b = Tensor([10, 20, 30], (1, 3), requires_grad=True)
	loss = (a + b).sum()
	loss.backward()
	_assert_list_close("grad a (2,1)", a.grad, [3, 3])
	_assert_list_close("grad b (1,3)", b.grad, [2, 2, 2])

	a = Tensor([1, 2, 3], (3,), requires_grad=True)
	b = Tensor([10, 20, 30, 40, 50, 60], (2, 3), requires_grad=True)
	loss = (a + b).sum()
	loss.backward()
	_assert_list_close("grad a (3,)", a.grad, [2, 2, 2])
	_assert_list_close("grad b (2,3)", b.grad, [1, 1, 1, 1, 1, 1])


def test_im2col():
	x = np.arange(1, 17, dtype=np.float64).reshape(1, 1, 4, 4)

	cols, meta = _im2col(x, kernel_size=3, stride=1, padding=0)

	assert cols.shape == (4, 9), f"expected (4, 9), got {cols.shape}"
	assert meta["out_H"] == 2 and meta["out_W"] == 2
	assert meta["N"] == 1 and meta["C"] == 1 and meta["H"] == 4 and meta["W"] == 4

	expected_patches = [
		[1, 2, 3, 5, 6, 7, 9, 10, 11],
		[2, 3, 4, 6, 7, 8, 10, 11, 12],
		[5, 6, 7, 9, 10, 11, 13, 14, 15],
		[6, 7, 8, 10, 11, 12, 14, 15, 16],
	]
	for i, expected in enumerate(expected_patches):
		_assert_list_close(f"im2col patch {i}", np.asarray(cols[i]), expected)


def test_conv2d_forward():
	layer = Conv2D(1, 2, kernel_size=3, stride=1, padding=0, bias=True)
	x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
	out = layer.forward(x)
	assert out.shape == (1, 2, 2, 2), f"expected (1, 2, 2, 2), got {out.shape}"


def test_conv2d_backward():
	np.random.seed(0)
	layer = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
	x = Tensor(np.arange(1, 10, dtype=np.float64), (1, 1, 3, 3), requires_grad=True)
	out = layer.forward(x)
	loss = out.sum()
	loss.backward()

	assert x.grad is not None
	assert layer.weights.grad is not None
	assert layer.biases.grad is not None
	assert len(x.grad) == 9
	assert len(layer.weights.grad) == 4
	assert len(layer.biases.grad) == 1


def test_maxpool2d_forward():
	layer = MaxPool2D(kernel_size=2, stride=2)
	x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
	out = layer.forward(x)
	assert out.shape == (1, 1, 2, 2)
	_assert_list_close("maxpool values", np.asarray(out.data), [6, 8, 14, 16])


def test_maxpool2d_backward():
	layer = MaxPool2D(kernel_size=2, stride=2)
	x = Tensor(np.arange(1, 17, dtype=np.float64), (1, 1, 4, 4), requires_grad=True)
	out = layer.forward(x)
	loss = out.sum()
	loss.backward()
	_assert_list_close("maxpool input grad", np.asarray(x.grad), [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1])


def test_flatten():
	layer = Flatten()
	x = Tensor(np.arange(24, dtype=np.float64), (2, 3, 2, 2), requires_grad=True)
	out = layer.forward(x)
	assert out.shape == (2, 12)
	loss = out.sum()
	loss.backward()
	assert x.grad is not None
	assert len(x.grad) == 24
	_assert_list_close("flatten grad", np.asarray(x.grad), np.ones(24))


def main():
	test_forward_broadcasting()
	test_backward_broadcasting()
	test_im2col()
	test_conv2d_forward()
	test_conv2d_backward()
	test_maxpool2d_forward()
	test_maxpool2d_backward()
	test_flatten()
	print("All tests passed.")


if __name__ == "__main__":
	main()
