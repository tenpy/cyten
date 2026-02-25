"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""
# Copyright (C) TeNPy Developers, Apache license

# Note: order matters to avoid circular imports!
# pyright: ignore
from .._core import Dtype  # noqa
from . import dtypes
from .dtypes import _DtypeEnumWrapper

from ._block_backend import Block, BlockBackend
from .array_api import ArrayApiBlockBackend
from .numpy import NumpyBlockBackend
from .torch import TorchBlockBackend

import numpy as np

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
