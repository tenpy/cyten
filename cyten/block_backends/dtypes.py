"""Utility function concerning dtypes and in particular the Dtype class."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import numpy as np

from .._core import Dtype

# Maps used by numpy / array backends. Dtype methods themselves are bound in C++.
_numpy_dtype_to_cyten = {
    None: None,
    np.bool_: Dtype.bool,
    np.float32: Dtype.float32,
    np.float64: Dtype.float64,
    np.complex64: Dtype.complex64,
    np.complex128: Dtype.complex128,
    np.dtype('bool'): Dtype.bool,
    np.dtype('float32'): Dtype.float32,
    np.dtype('float64'): Dtype.float64,
    np.dtype('complex64'): Dtype.complex64,
    np.dtype('complex128'): Dtype.complex128,
}

_cyten_dtype_to_numpy = {
    None: None,
    Dtype.bool: np.bool_,
    Dtype.float32: np.dtype('float32'),
    Dtype.float64: np.dtype('float64'),
    Dtype.complex64: np.dtype('complex64'),
    Dtype.complex128: np.dtype('complex128'),
}
