"""See :mod:`cyten.symmetries`"""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import numpy as np

# implemented in C++
from .._core import (
    SU2,  # noqa: F401
    SUN,  # noqa: F401
    U1,  # noqa: F401
    ZN,  # noqa: F401
    AbelianGroup,  # noqa: F401
    BaseSymmetry,  # noqa: F401
    BraidChiralityUnspecifiedError,  # noqa: F401
    BraidingStyle,  # noqa: F401
    FermionNumber,  # noqa: F401
    FermionParity,  # noqa: F401
    FibonacciAnyonCategory,  # noqa: F401
    FusionStyle,  # noqa: F401
    Group,  # noqa: F401
    IsingAnyonCategory,  # noqa: F401
    NoSymmetry,  # noqa: F401
    QuantumDoubleZNAnyonCategory,  # noqa: F401
    Sector,  # noqa: F401
    SectorArray,  # noqa: F401
    SU2_kAnyonCategory,  # noqa: F401
    SU3_3AnyonCategory,  # noqa: F401
    Symmetry,  # noqa: F401
    SymmetryError,  # noqa: F401
    SymmetryFactor,  # noqa: F401
    ToricCodeCategory,  # noqa: F401
    ZNAnyonCategory,  # noqa: F401
    ZNAnyonCategory2,  # noqa: F401
    double_semion_category,  # noqa: F401
    semion_category,  # noqa: F401
)
from .sector_utils import (  # noqa: F401
    as_sector,
    as_sector_array,
    assert_sectors_equal,
    iter_common_sorted_sector_arrays,
)

try:
    import h5py

    h5py_version = h5py.version.version_tuple
except (ImportError, AttributeError):  # fmt: skip
    h5py_version = (0, 0)


def _default_c_symbol(sym, a, b, c, d, e, f):
    """C-symbol from R and F symbols (same formula as ``BaseSymmetry._c_symbol``).

    Prefer this over ``super()._c_symbol`` when the subclass overrides ``_c_symbol``:
    with a C++ / pybind trampoline base, ``super()`` re-enters the Python override.
    """
    R1 = sym._r_symbol(e, c, d)
    F = sym._f_symbol(c, a, b, d, e, f)
    R2 = sym._r_symbol(a, c, f)
    return R1.reshape(1, -1, 1, 1) * F * np.conj(R2).reshape(1, 1, -1, 1)
