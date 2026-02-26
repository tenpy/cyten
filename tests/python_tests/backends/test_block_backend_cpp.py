"""Minimal tests for C++ BlockBackend / NumpyBlockBackend from cyten._core."""

from __future__ import annotations

import numpy as np
from cyten._core import Dtype, NumpyBlockBackend


def test_numpy_block_backend_zeros_get_shape():
    be = NumpyBlockBackend()
    z = be.zeros([2, 3], Dtype.float64)
    assert be.get_shape(z) == [2, 3]
    assert be.get_dtype(z) == Dtype.float64
    assert be.get_device(z) == 'cpu'


def test_numpy_block_backend_copy_block():
    be = NumpyBlockBackend()
    z = be.zeros([2, 2], Dtype.float64)
    c = be.copy_block(z)
    assert be.get_shape(c) == [2, 2]
    arr = be.to_numpy(c)
    np.testing.assert_array_equal(arr, np.zeros((2, 2)))


def test_numpy_block_backend_apply_leg_permutations():
    be = NumpyBlockBackend()
    # block shape (2, 3); permute first axis [1,0], second axis identity [0,1,2]
    z = be.zeros([2, 3], Dtype.float64)
    arr = be.to_numpy(z)
    arr[0, 0] = 1.0
    z = be.as_block(arr)
    perms = [np.array([1, 0], dtype=np.int64), np.array([0, 1, 2], dtype=np.int64)]
    p = be.apply_leg_permutations(z, perms)
    assert be.get_shape(p) == [2, 3]
    out = be.to_numpy(p)
    # row 0 and row 1 swapped: (1,0,0) moved to row 1
    assert out[1, 0] == 1.0
    assert out[0, 0] == 0.0
