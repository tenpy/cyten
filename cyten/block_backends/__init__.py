"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""
# Copyright (C) TeNPy Developers, Apache license

# Note: order matters to avoid circular imports!
# pyright: ignore
from .._core import Dtype  # noqa
from .._core import NumpyBlock, NumpyBlockBackend as _NumpyBlockBackendCpp
from . import dtypes
from .dtypes import _DtypeEnumWrapper

from ._block_backend import Block, BlockBackend
from .array_api import ArrayApiBlockBackend
from .torch import TorchBlockBackend

import numpy as np


class NumpyBlockBackend(_NumpyBlockBackendCpp):
    """Numpy block backend (C++ implementation)."""

    BlockCls = (np.ndarray, NumpyBlock)

    def permute_combined_matrix(self, block, dims1, idcs1, dims2, idcs2):
        dims1, idcs1, dims2, idcs2 = list(dims1), list(idcs1), list(dims2), list(idcs2)
        input_was_ndarray = isinstance(block, np.ndarray)
        if isinstance(block, np.ndarray):
            block = _NumpyBlockBackendCpp.as_block(self, block, None, None)
        res = _NumpyBlockBackendCpp.permute_combined_matrix(self, block, dims1, idcs1, dims2, idcs2)
        return res.array() if input_was_ndarray else res

    def permute_combined_idx(self, block, axis, dims, idcs):
        dims, idcs = list(dims), list(idcs)
        input_was_ndarray = isinstance(block, np.ndarray)
        if isinstance(block, np.ndarray):
            block = _NumpyBlockBackendCpp.as_block(self, block, None, None)
        res = _NumpyBlockBackendCpp.permute_combined_idx(self, block, axis, dims, idcs)
        return res.array() if input_was_ndarray else res

    def test_block_sanity(
        self,
        block,
        expect_shape=None,
        expect_dtype=None,
        expect_device=None,
    ):
        if not isinstance(block, self.BlockCls):
            raise AssertionError('wrong block type')
        # Convert ndarray to NumpyBlock so C++ can validate (C++ throws RuntimeError on failure)
        if isinstance(block, np.ndarray):
            block = _NumpyBlockBackendCpp.as_block(self, block, None, None)
        _NumpyBlockBackendCpp.test_block_sanity(self, block, expect_shape, expect_dtype, expect_device)


dtypes.Dtype = Dtype
dtypes._cyten_dtype_to_numpy[Dtype.bool] = np.bool_
dtypes._cyten_dtype_to_numpy[Dtype.float32] = np.dtype('float32')
dtypes._cyten_dtype_to_numpy[Dtype.float64] = np.dtype('float64')
dtypes._cyten_dtype_to_numpy[Dtype.complex64] = np.dtype('complex64')
dtypes._cyten_dtype_to_numpy[Dtype.complex128] = np.dtype('complex128')
dtypes._numpy_dtype_to_cyten[np.bool_] = Dtype.bool
dtypes._numpy_dtype_to_cyten[np.float32] = Dtype.float32
dtypes._numpy_dtype_to_cyten[np.float64] = Dtype.float64
dtypes._numpy_dtype_to_cyten[np.complex64] = Dtype.complex64
dtypes._numpy_dtype_to_cyten[np.complex128] = Dtype.complex128
dtypes._numpy_dtype_to_cyten[np.dtype('bool')] = Dtype.bool
dtypes._numpy_dtype_to_cyten[np.dtype('float32')] = Dtype.float32
dtypes._numpy_dtype_to_cyten[np.dtype('float64')] = Dtype.float64
dtypes._numpy_dtype_to_cyten[np.dtype('complex64')] = Dtype.complex64
dtypes._numpy_dtype_to_cyten[np.dtype('complex128')] = Dtype.complex128
