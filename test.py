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


def main():
	test_forward_broadcasting()
	test_backward_broadcasting()
	print("Broadcasting tests passed.")


if __name__ == "__main__":
	main()
