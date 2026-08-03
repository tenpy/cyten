"""Minimal tests for C++ BlockBackend / NumpyBlockBackend from cyten._core."""

from __future__ import annotations

import numpy as np
from cyten._core import Dtype, NumpyBlockBackend


def test_numpy_block_backend_zeros_get_shape():
    be = NumpyBlockBackend.from_factory('cpu')
    z = be.zeros([2, 3], Dtype.float64)
    assert be.get_shape(z) == (2, 3)
    assert be.get_dtype(z) == Dtype.float64
    assert be.get_device(z) == 'cpu'


def test_numpy_block_backend_copy_block():
    be = NumpyBlockBackend.from_factory('cpu')
    z = be.zeros([2, 2], Dtype.float64)
    c = be.copy_block(z)
    assert be.get_shape(c) == (2, 2)
    arr = be.to_numpy(c)
    np.testing.assert_array_equal(arr, np.zeros((2, 2)))


def test_block_getitem_setitem_scalar_by_int_indices():
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    block = be.as_block(np.arange(6, dtype=np.float64).reshape(2, 3))
    # integer multi-index prefers Scalar getitem
    s = block[0, 2]
    assert isinstance(s, BlockBackend.Scalar)
    assert s.as_float64() == 2.0
    s_list = block[[1, 0]]
    assert isinstance(s_list, BlockBackend.Scalar)
    assert s_list.as_float64() == 3.0
    # Scalar setitem by integer multi-index
    block[1, 2] = be.as_scalar(42.0)
    assert block[1, 2].as_float64() == 42.0
    # slices still return a Block
    sliced = block[0, :]
    assert isinstance(sliced, BlockBackend.BlockCls)
    assert tuple(sliced.shape) == (3,)


def test_block_getitem_setitem_scalar_1d():
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    vec = be.as_block(np.array([10.0, 20.0, 30.0]))
    s = vec[1]
    assert isinstance(s, BlockBackend.Scalar)
    assert s.as_float64() == 20.0
    vec[2] = be.as_scalar(99.0)
    assert vec[2].as_float64() == 99.0
    # slice still returns a Block
    sliced = vec[1:]
    assert isinstance(sliced, BlockBackend.BlockCls)
    assert tuple(sliced.shape) == (2,)


def test_block_getitem_setitem_slice_and_index_array():
    """Advanced indexing (slices / index arrays) returns Blocks and supports setitem."""
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    block = be.as_block(np.arange(12, dtype=np.float64).reshape(3, 4))

    row = block[1, :]
    assert isinstance(row, BlockBackend.BlockCls)
    np.testing.assert_array_equal(be.to_numpy(row), np.arange(4, 8, dtype=np.float64))

    col = block[:, 2]
    assert isinstance(col, BlockBackend.BlockCls)
    np.testing.assert_array_equal(be.to_numpy(col), np.array([2.0, 6.0, 10.0]))

    # integer index array on one axis
    sub = block[[2, 0], :]
    assert isinstance(sub, BlockBackend.BlockCls)
    np.testing.assert_array_equal(be.to_numpy(sub), np.array([[8.0, 9.0, 10.0, 11.0], [0.0, 1.0, 2.0, 3.0]]))

    # setitem via slice
    block[0, 1:3] = be.as_block(np.array([7.0, 8.0]))
    np.testing.assert_array_equal(be.to_numpy(block)[0, 1:3], np.array([7.0, 8.0]))


def test_block_abs():
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    block = be.as_block(np.array([-1.0, 2.0, -3.0]))
    out = abs(block)
    assert isinstance(out, BlockBackend.BlockCls)
    np.testing.assert_array_equal(be.to_numpy(out), np.array([1.0, 2.0, 3.0]))
    # complex: magnitude
    cblock = be.as_block(np.array([3.0 + 4.0j, -1.0]))
    cout = abs(cblock)
    np.testing.assert_allclose(be.to_numpy(cout), np.array([5.0, 1.0]))


def test_scalar_abs():
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    s = abs(be.as_scalar(-2.5))
    assert isinstance(s, BlockBackend.Scalar)
    assert s.as_float64() == 2.5
    cs = abs(be.as_scalar(3.0 + 4.0j))
    assert isinstance(cs, BlockBackend.Scalar)
    assert cs.as_float64() == 5.0


def test_scalar_sqrt_exp_log_pow():
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    s = be.as_scalar(4.0)
    assert isinstance(s.sqrt(), BlockBackend.Scalar)
    assert s.sqrt().as_float64() == 2.0
    assert s.exp().as_float64() == np.exp(4.0)
    assert s.log().as_float64() == np.log(4.0)
    assert (s**2).as_float64() == 16.0
    assert (s ** be.as_scalar(0.5)).as_float64() == 2.0
    assert s.pow(be.as_scalar(3.0)).as_float64() == 64.0


def test_scalar_real_imag():
    from cyten._core import BlockBackend

    be = NumpyBlockBackend.from_factory('cpu')
    z = be.as_scalar(3.0 + 4.0j)
    r = z.real()
    assert isinstance(r, BlockBackend.Scalar)
    assert r.as_float64() == 3.0
    assert r.dtype == Dtype.float64
    i = z.imag()
    assert isinstance(i, BlockBackend.Scalar)
    assert i.as_float64() == 4.0
    assert i.dtype == Dtype.float64


def test_numpy_block_backend_apply_leg_permutations():
    be = NumpyBlockBackend.from_factory('cpu')
    # block shape (2, 3); permute first axis [1,0], second axis identity [0,1,2]
    z = be.zeros([2, 3], Dtype.float64)
    arr = be.to_numpy(z)
    arr[0, 0] = 1.0
    z = be.as_block(arr)
    perms = [np.array([1, 0], dtype=np.int64), np.array([0, 1, 2], dtype=np.int64)]
    p = be.apply_leg_permutations(z, perms)
    assert be.get_shape(p) == (2, 3)
    out = be.to_numpy(p)
    # row 0 and row 1 swapped: (1,0,0) moved to row 1
    assert out[1, 0] == 1.0
    assert out[0, 0] == 0.0
