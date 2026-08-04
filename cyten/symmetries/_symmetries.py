"""See :mod:`cyten.symmetries`"""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import numpy as np
from numpy import typing as npt

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
    SU2_kAnyonCategory,  # noqa: F401
    SU3_3AnyonCategory,  # noqa: F401
    Symmetry,  # noqa: F401
    SymmetryError,  # noqa: F401
    SymmetryFactor,  # noqa: F401
    ToricCodeCategory,  # noqa: F401
    ZNAnyonCategory,  # noqa: F401
    ZNAnyonCategory2,  # noqa: F401
)
from ..tools.misc import as_immutable_array

try:
    import h5py

    h5py_version = h5py.version.version_tuple
except (ImportError, AttributeError):
    h5py_version = (0, 0)


# these are the known results for e.g. N symbols, F symbols, ... in some special cases
one_1D = as_immutable_array(np.ones((1), dtype=int))
one_2D = as_immutable_array(np.ones((1, 1), dtype=int))
one_2D_float = as_immutable_array(np.ones((1, 1), dtype=float))
one_4D = as_immutable_array(np.ones((1, 1, 1, 1), dtype=int))
one_4D_float = as_immutable_array(np.ones((1, 1, 1, 1), dtype=float))


def _default_c_symbol(sym, a, b, c, d, e, f):
    """C-symbol from R and F symbols (same formula as ``BaseSymmetry._c_symbol``).

    Prefer this over ``super()._c_symbol`` when the subclass overrides ``_c_symbol``:
    with a C++ / pybind trampoline base, ``super()`` re-enters the Python override.
    """
    R1 = sym._r_symbol(e, c, d)
    F = sym._f_symbol(c, a, b, d, e, f)
    R2 = sym._r_symbol(a, c, f)
    return R1.reshape(1, -1, 1, 1) * F * np.conj(R2).reshape(1, 1, -1, 1)


Sector = npt.NDArray[np.int_]
"""Type hint for a sector. A 1D array of integers with axis [q] and shape ``(sector_ind_len,)``."""

SectorArray = npt.NDArray[np.int_]
"""Type hint for an array of multiple sectors.

A 2D array of int with axis [s, q] and shape ``(num_sectors, sector_ind_len)``.
"""


def _Symmetry_from_hdf5(cls, hdf5_loader, h5gr, subpath):
    """Reconstruct :class:`Symmetry` from HDF5 (C++ has :meth:`save_hdf5` only)."""
    factors = hdf5_loader.load(subpath + 'factors')
    obj = cls(factors)
    hdf5_loader.memorize_load(h5gr, obj)
    return obj


Symmetry.from_hdf5 = classmethod(_Symmetry_from_hdf5)


no_symmetry = NoSymmetry().as_Symmetry()
z2_symmetry = ZN(N=2).as_Symmetry()
z3_symmetry = ZN(N=3).as_Symmetry()
z4_symmetry = ZN(N=4).as_Symmetry()
z5_symmetry = ZN(N=5).as_Symmetry()
z6_symmetry = ZN(N=6).as_Symmetry()
z7_symmetry = ZN(N=7).as_Symmetry()
z8_symmetry = ZN(N=8).as_Symmetry()
z9_symmetry = ZN(N=9).as_Symmetry()
u1_symmetry = U1().as_Symmetry()
su2_symmetry = SU2().as_Symmetry()
fermion_number = FermionNumber().as_Symmetry()
fermion_parity = FermionParity().as_Symmetry()
semion_category = ZNAnyonCategory2(2, 0).as_Symmetry()
toric_code_category = ToricCodeCategory().as_Symmetry()
double_semion_category = ZNAnyonCategory2(2, 0) * ZNAnyonCategory2(2, 1)
fibonacci_anyon_category = FibonacciAnyonCategory(handedness='left').as_Symmetry()
ising_anyon_category = IsingAnyonCategory(nu=1).as_Symmetry()
