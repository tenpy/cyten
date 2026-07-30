"""Smoke tests for C++ ArrayApiBlockBackend (numpy as Array API namespace)."""

import numpy as np
import pytest

from cyten.block_backends import ArrayApiBlockBackend, Dtype


@pytest.fixture
def arrayapi_backend():
    return ArrayApiBlockBackend(np, default_device='cpu')


def test_zeros_and_item(arrayapi_backend):
    bb = arrayapi_backend
    z = bb.zeros([2, 3], Dtype.float64)
    assert bb.get_shape(z) == (2, 3)
    assert bb.get_dtype(z) == Dtype.float64
    assert float(bb.sum_all(z).as_float64()) == 0.0


def test_outer_and_tdot(arrayapi_backend):
    bb = arrayapi_backend
    a = bb.as_block([[1.0, 2.0]], Dtype.float64)
    b = bb.as_block([3.0, 4.0], Dtype.float64)
    o = bb.outer(a, b)
    assert bb.get_shape(o) == (1, 2, 2)
    t = bb.tdot(a, b, [1], [0])
    assert bb.get_shape(t) == (1,)
    assert float(bb.item(t).as_float64()) == pytest.approx(1 * 3 + 2 * 4)


def test_python_subclass_override_kron():
    class MyArrayApi(ArrayApiBlockBackend):
        def kron(self, a, b):
            # Delegate via numpy for the smoke test.
            return self.block_from_numpy(np.kron(a.to_numpy(), b.to_numpy()))

    be = MyArrayApi(np, default_device='cpu')
    a = be.as_block([[1.0, 0.0], [0.0, 1.0]], Dtype.float64)
    b = be.as_block([[2.0]], Dtype.float64)
    k = be.kron(a, b)
    np.testing.assert_allclose(k.to_numpy(), np.kron([[1, 0], [0, 1]], [[2]]))
