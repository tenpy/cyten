"""Tests for :class:`~cyten.block_backends.dtypes.Dtype`."""

import numpy as np
import pytest

from cyten import Dtype
from cyten.tools import hdf5_io

h5py = pytest.importorskip('h5py')

# Expected numpy scalar types returned by Dtype.to_numpy_dtype().
_EXPECTED_NUMPY = {
    Dtype.bool: np.bool_,
    Dtype.float32: np.float32,
    Dtype.float64: np.float64,
    Dtype.complex64: np.complex64,
    Dtype.complex128: np.complex128,
    Dtype.int64: np.int64,
}


@pytest.mark.parametrize('dtype, expected_np', list(_EXPECTED_NUMPY.items()), ids=lambda x: getattr(x, 'name', str(x)))
def test_to_numpy_dtype_and_back(dtype, expected_np):
    numpy_dtype = dtype.to_numpy_dtype()
    assert numpy_dtype is expected_np or numpy_dtype == expected_np
    assert Dtype.from_numpy_dtype(numpy_dtype) is dtype
    # Also accept np.dtype(...) wrappers
    assert Dtype.from_numpy_dtype(np.dtype(numpy_dtype)) is dtype


@pytest.mark.parametrize(
    'dtype, expect_real',
    [
        (Dtype.bool, True),
        (Dtype.int64, True),
        (Dtype.float32, True),
        (Dtype.float64, True),
        (Dtype.complex64, False),
        (Dtype.complex128, False),
    ],
    ids=lambda x: getattr(x, 'name', str(x)),
)
def test_is_real(dtype, expect_real):
    assert dtype.is_real is expect_real
    assert dtype.is_complex is (not expect_real)


@pytest.mark.parametrize(
    'dtype, expect',
    [
        (Dtype.float32, Dtype.float32),
        (Dtype.float64, Dtype.float64),
        (Dtype.complex64, Dtype.float32),
        (Dtype.complex128, Dtype.float64),
    ],
    ids=lambda x: getattr(x, 'name', str(x)),
)
def test_to_real(dtype, expect):
    assert dtype.to_real is expect


@pytest.mark.parametrize('dtype', [Dtype.bool, Dtype.int64], ids=lambda d: d.name)
def test_to_real_rejects_bool_and_int64(dtype):
    with pytest.raises(ValueError, match='can not be converted to real'):
        _ = dtype.to_real


@pytest.mark.parametrize(
    'dtypes, expect',
    [
        ((Dtype.float32, Dtype.float64), Dtype.float64),
        ((Dtype.float64, Dtype.complex64), Dtype.complex128),
        ((Dtype.float32, Dtype.complex64), Dtype.complex64),
        ((Dtype.complex64, Dtype.float32), Dtype.complex64),
        ((Dtype.float32, Dtype.float64, Dtype.complex64), Dtype.complex128),
        ((Dtype.bool, Dtype.float32), Dtype.float32),
    ],
    ids=lambda x: '+'.join(d.name for d in x) if isinstance(x, tuple) else str(x),
)
def test_common(dtypes, expect):
    assert Dtype.common(*dtypes) is expect
    # Also as instance method: first.dtype.common(*rest)
    first, *rest = dtypes
    assert first.common(*rest) is expect


@pytest.mark.parametrize('dtype', list(Dtype), ids=lambda d: d.name)
def test_dtype_hdf5_roundtrip(dtype, tmp_path):
    filename = tmp_path / f'dtype_{dtype.name}.hdf5'
    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, dtype)
    with h5py.File(str(filename), 'r') as f:
        loaded = hdf5_io.load_from_hdf5(f)
    assert loaded is dtype
    assert loaded == dtype
