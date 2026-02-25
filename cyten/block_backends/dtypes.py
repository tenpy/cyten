"""Utility function concerning dtypes and in particular the Dtype class."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from enum import Enum
from numbers import Number


class _DtypeEnumWrapper(Enum):
    """The dtype of (entries in) a tensor."""

    # value = num_bytes * 2 + int(not is_real)
    # enum entries are defined in C++ include/cyten/dtypes.h and mapped in pybind/dtypes.cpp
    # bool = 2
    # float32 = 8
    # complex64 = 9
    # float64 = 16
    # complex128 = 17

    @property
    def is_real(dtype):
        return dtype.value % 2 == 0

    @property
    def is_complex(dtype):
        return dtype.value % 2 == 1

    @property
    def to_complex(dtype):
        if dtype.value == 2:
            raise ValueError('Dtype.bool can not be converted to complex')
        if dtype.value % 2 == 1:
            return dtype
        return Dtype(dtype.value + 1)  # noqa: F821

    @property
    def to_real(dtype):
        if dtype.value == 2:
            raise ValueError('Dtype.bool can not be converted to real')
        if dtype.value % 2 == 0:
            return dtype
        return Dtype(dtype.value - 1)  # noqa: F821

    @property
    def python_type(dtype):
        if dtype.value == 2:
            return bool
        if dtype.is_real:
            return float
        return complex

    @property
    def zero_scalar(dtype):
        return dtype.python_type(0)

    @property
    def eps(dtype):
        # difference between 1.0 and the next representable floating point number at the given precision
        if dtype.value == Dtype.bool.value:  # noqa: F821
            raise ValueError(f'{dtype} is not inexact')
        n_bits = 8 * (dtype.value // 2)
        if n_bits == 32:
            return 2**-23
        if n_bits == 64:
            return 2**-52
        raise NotImplementedError(f'Dtype.eps not implemented for n_bits={n_bits}')

    def __repr__(self) -> str:
        return f'Dtype.{self.name}'

    def common(*dtypes):
        res = Dtype(max(t.value for t in dtypes))  # noqa: F821
        if res.is_real:
            if not all(t.is_real for t in dtypes):
                return res.to_complex
        return res

    def convert_python_scalar(dtype, value) -> complex | float | bool:
        if dtype.value == Dtype.bool.value:  # noqa: F821
            if value in [True, False, 0, 1]:
                return bool(value)
        elif dtype.is_real:
            if isinstance(value, (int, float)):
                return float(value)
        else:
            if isinstance(value, Number):
                return complex(value)
        raise TypeError(f'Type {type(value)} is incompatible with dtype {dtype}')

    def to_numpy_dtype(dtype):
        return _cyten_dtype_to_numpy[dtype]

    @classmethod
    def from_numpy_dtype(cls, dtype):
        return _numpy_dtype_to_cyten[dtype]


_numpy_dtype_to_cyten = {None: None}
# actual values are filled in __init__.py after import of _core, since we need the values from C++
#     np.float32: Dtype.float32,
#     np.float64: Dtype.float64,
#     np.complex64: Dtype.complex64,
#     np.complex128: Dtype.complex128,
#     np.bool_: Dtype.bool,
#     np.dtype('float32'): Dtype.float32,
#     np.dtype('float64'): Dtype.float64,
#     np.dtype('complex64'): Dtype.complex64,
#     np.dtype('complex128'): Dtype.complex128,
#     np.dtype('bool'): Dtype.bool,


_cyten_dtype_to_numpy = {None: None}
#     Dtype.float32: np.float32,
#     Dtype.float64: np.float64,
#     Dtype.complex64: np.complex64,
#     Dtype.complex128: np.complex128,
#     Dtype.bool: np.bool_,
