"""Minimal torch block-backend smoke tests."""

import numpy as np
import pytest

from cyten.backends.backend_factory import get_backend
from cyten.block_backends import TorchBlockBackend
from cyten.block_backends.dtypes import Dtype
from cyten.symmetries import no_symmetry
from cyten.tensors import SymmetricTensor
from cyten.testing import random_tensor


@pytest.mark.torch
def test_torch_block_backend_zeros():
    bb = TorchBlockBackend.from_factory('cpu:0')
    z = bb.zeros([2, 3], Dtype.float64)
    assert bb.get_shape(z) == (2, 3)
    assert bb.get_dtype(z) == Dtype.float64
    assert bb.get_device(z) == 'cpu:0'
    assert float(bb.sum_all(z).as_float64()) == 0.0


@pytest.mark.torch
def test_to_backend_numpy_to_torch():
    np_random = np.random.default_rng(0)
    b1 = get_backend('no_symmetry', 'numpy')
    b2 = get_backend('no_symmetry', 'torch')
    tens = random_tensor(no_symmetry, 1, 1, backend=b1, np_random=np_random, cls=SymmetricTensor)
    res = tens.to_backend(b2)
    res.test_sanity()
    assert isinstance(res.backend.block_backend, TorchBlockBackend)
    recovered = res.to_backend(b1)
    recovered.test_sanity()
