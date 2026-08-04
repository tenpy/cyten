"""See :mod:`cyten.symmetries`"""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import math
from itertools import product
from typing import Literal

import numpy as np
from numpy import typing as npt

# implemented in C++
from .._core import (
    U1,  # noqa: F401
    ZN,  # noqa: F401
    AbelianGroup,  # noqa: F401
    BaseSymmetry,  # noqa: F401
    BraidChiralityUnspecifiedError,  # noqa: F401
    BraidingStyle,  # noqa: F401
    FusionStyle,  # noqa: F401
    Group,  # noqa: F401
    NoSymmetry,  # noqa: F401
    SU2,  # noqa: F401
    SUN,  # noqa: F401
    Symmetry,  # noqa: F401
    SymmetryError,  # noqa: F401
    SymmetryFactor,  # noqa: F401
)
from ..block_backends.dtypes import Dtype
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


class FermionNumber(SymmetryFactor):
    """Conserves a fermionic particle number.

    .. warning ::
        A symmetry that conserves the individual particle numbers of multiple fermion species
        is *not* given by a product of :class:`FermionNumber` symmetries!
        This is because it would not reproduce the physically relevant braiding, as the different
        species would then behave as mutual *bosons* (i.e. braiding an A-type fermion with a B-type
        fermion would not give a sign).
        Instead, you should form a product symmetry where each particle number is covered by a
        :class:`U1Symmetry` factor (one per species with conserved particle number), while the
        fermionic statistics is covered by an extra factor of :class:`FermionParity`.

    This is essentially U(1), but with a braid that encodes fermionic exchange statistics.
    Allowed sectors are arrays with a single integer entry.
    """

    fusion_tensor_dtype = Dtype.float64

    def __init__(self, descriptive_name: str = None, trivial_shift: bool = True):
        super().__init__(
            fusion_style=FusionStyle.single,
            braiding_style=BraidingStyle.fermionic,
            trivial_sector=np.array([0], int),
            group_name='FermionNumber',
            num_sectors=np.inf,
            has_complex_topological_data=False,
            descriptive_name=descriptive_name,
            trivial_shift=trivial_shift,
        )

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,)

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return a + b

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors)

    def sector_dim(self, a):
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def _is_equivalent_factor(self, other):
        return isinstance(other, FermionNumber)

    def dual_sector(self, a: Sector) -> Sector:
        return -a

    def dual_sectors(self, sectors):
        return -sectors

    def _n_symbol(self, a, b, c):
        return 1

    def _f_symbol(self, a, b, c, d, e, f):
        return one_4D

    def frobenius_schur(self, a):
        return 1

    def qdim(self, a):
        return 1

    def sqrt_qdim(self, a):
        return 1

    def inv_sqrt_qdim(self, a):
        return 1

    def _b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # sqrt(d_b) [F^{a b dual(b)}_a]^{111}_{c,mu,nu} = sqrt(1) * 1 = 1
        return one_2D

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # if a and b are odd -1, otherwise +1
        # in the first (second) case above, we have ``a * b`` equal to 1 (0).
        return 1 - 2 * np.mod(a, 2) * np.mod(b, 2)

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        # F = 1  -->  C = R^{ec}_d conj(R)^{ca}_f
        C = (1 - 2 * np.mod(e, 2) * np.mod(c, 2)) * (1 - 2 * np.mod(c, 2) * np.mod(a, 2))
        return C[None, None, None, :]

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool, Z_b: bool) -> np.ndarray:
        return one_4D_float

    def swap_gate(self, a: Sector, b: Sector):
        # if a and b are odd -1, otherwise +1
        # in the first (second) case above, we have ``a * b`` equal to 1 (0).
        sign = 1 - 2 * np.mod(a, 2) * np.mod(b, 2)
        return sign * one_4D_float

    def topological_twist(self, a):
        # +1 for even parity, -1 for odd
        return 1 - 2 * np.mod(a, 2).item()

    def Z_iso(self, a):
        return one_2D_float

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'FermionNumber({name_str})'


class FermionParity(SymmetryFactor):
    """Fermionic Parity.

    .. warning ::
        A symmetry that conserves the individual particle number parities of multiple fermion
        species is *not* given by a product of :class:`FermionParity` symmetries!
        This is because it would not reproduce the physically relevant braiding, as the different
        species would then behave as mutual *bosons* (i.e. braiding an A-type fermion with a B-type
        fermion would not give a sign).
        Instead, you should form a product symmetry where each particle number parity is covered by
        a :class:`ZNSymmetry` factor (one per species with individually conserved parity), while the
        fermionic statistics is covered by an extra factor of :class:`FermionParity`.

    Allowed sectors are arrays with a single entry; either ``[0]`` (even) or ``1`` (odd).
    The parity is the number of fermions in a given state modulo 2.
    """

    fusion_tensor_dtype = Dtype.float64
    even = as_immutable_array(np.array([0], dtype=int))
    odd = as_immutable_array(np.array([1], dtype=int))

    def __init__(self, descriptive_name: str = None, trivial_shift: bool = True):
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.single,
            braiding_style=BraidingStyle.fermionic,
            trivial_sector=np.array([0], dtype=int),
            group_name='FermionParity',
            num_sectors=2,
            has_complex_topological_data=False,
            descriptive_name=descriptive_name,
            trivial_shift=trivial_shift,
        )

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 2

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 2)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        # equal sectors fuse to even parity, i.e. to `0 == (0 + 0) % 2 == (1 + 1) % 2`
        # unequal sectors fuse to odd parity i.e. to `1 == (0 + 1) % 2 == (1 + 0) % 2`
        return (a + b) % 2

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % 2

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def sector_str(self, a: Sector) -> str:
        return 'even' if a[0] == 0 else 'odd'

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'FermionParity({name_str})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, FermionParity)

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def sqrt_qdim(self, a: Sector) -> float:
        return 1

    def inv_sqrt_qdim(self, a: Sector) -> float:
        return 1

    def _b_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # sqrt(d_b) [F^{a b dual(b)}_a]^{111}_{c,mu,nu} = sqrt(1) * 1 = 1
        return one_2D

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        # if a and b are fermionic -1, otherwise +1
        # in the first (second) case above, we have ``a * b`` equal to 1 (0).
        return 1 - 2 * a * b

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        # R^{ec}_d conj(R)^{ca}_f
        C = (1 - 2 * e * c) * (1 - 2 * c * a)
        return C[None, None, None, :]

    def _fusion_tensor(self, a: Sector, b: Sector, c: Sector, Z_a: bool, Z_b: bool) -> np.ndarray:
        return one_4D_float

    def swap_gate(self, a: Sector, b: Sector):
        # if a and b are fermionic -1, otherwise +1
        # in the first (second) case above, we have ``a * b`` equal to 1 (0).
        sign = 1 - 2 * a * b
        return sign * one_4D_float

    def topological_twist(self, a):
        return 1 - 2 * a.item()

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]

    def Z_iso(self, a: Sector) -> np.ndarray:
        return one_2D_float


class ZNAnyonCategory(SymmetryFactor):
    r"""Abelian anyon category with fusion rules corresponding to the Z_N group;

    also written as :math:`Z_N^{(n)}`.

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`

    While `N` determines number of anyons, `n` determines the R-symbols, i.e., the exchange
    statistics. Since `n` and `n+N` describe the same statistics, :math:`n \in Z_N`.
    Reduces to the Z_N abelian group symmetry for `n = 0`. Use `ZNSymmetry` for this case!

    The anyon category corresponding to opposite handedness is obtained for `N` and `N-n` (or `-n`).
    """

    def __init__(self, N: int, n: int, descriptive_name: str | None = None):
        assert isinstance(N, int)
        assert N > 1
        assert isinstance(n, int)
        self.N = N
        self.n = n = n % N
        self._phase = np.exp(2j * np.pi * n / N)
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.single,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0], dtype=int),
            group_name=f'ℤ_{N}^{n} anyon category',
            num_sectors=N,
            has_complex_topological_data=n > 0,
            descriptive_name=descriptive_name,
        )

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a[0] < self.N

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'ZNAnyonCategory({self.N}, {self.n}, {name_str})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, ZNAnyonCategory) and other.N == self.N and other.n == self.n

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._phase ** (a * b)

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._phase ** (b[0] * c[0]) * one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class ZNAnyonCategory2(SymmetryFactor):
    r"""Abelian anyon category with fusion rules corresponding to the Z_N group;

    also written as :math:`Z_N^{(n+1/2)}`. `N` must be even.

    Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
    `[0]`, `[1]`, ..., `[N-1]`

    While `N` determines number of anyons, `n` determines the R-symbols, i.e., the exchange
    statistics. Since `n` and `n+N` describe the same statistics, :math:`n \in Z_N`.
    Reduces to the Z_N abelian group symmetry for `n = 0`. Use `ZNSymmetry` for this case!

    The anyon category corresponding to opposite handedness is obtained for `N` and `N-n` (or `-n`).
    """

    def __init__(self, N: int, n: int, descriptive_name: str | None = None):
        assert isinstance(N, int)
        assert N > 1
        assert N % 2 == 0
        assert isinstance(n, int)
        self.N = N
        self.n = n % N
        self._phase = np.exp(2j * np.pi * (self.n + 0.5) / self.N)
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.single,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0], dtype=int),
            group_name=f'ℤ_{N}^({n}+1/2) anyon category',
            num_sectors=N,
            has_complex_topological_data=True,
            descriptive_name=descriptive_name,
        )

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < self.N

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'ZNAnyonCategory2({self.N}, {self.n}, {name_str})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, ZNAnyonCategory2) and other.N == self.N and other.n == self.n

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return (-1) ** (a[0] * ((b[0] + c[0]) // self.N)) * one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return (-1) ** a[0]

    def qdim(self, a: Sector) -> float:
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._phase ** (a * b) * one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return (self._phase ** (b[0] * c[0])) * one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.N, dtype=int)[:, None]


class QuantumDoubleZNAnyonCategory(SymmetryFactor):
    r"""Doubled abelian anyon category.

    The fusion rules corresponding to the :math:`Z_N \times Z_N` group.
    The category is commonly written as :math:`D(Z_N)`.

    Allowed sectors are 1D arrays with two integers between ``0`` and ``N-1``.
    ``[0, 0]``, ``[0, 1]``, ..., ``[N-1, N-1]``.

    This is not a simple product of two :class:`ZNAnyonCategory`\ s; there are nontrivial R-symbols.
    """

    def __init__(self, N: int, descriptive_name: str | None = None):
        assert isinstance(N, int)
        assert N > 1
        self.N = N
        self._phase = np.exp(2j * np.pi / self.N)
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.single,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0, 0], dtype=int),
            group_name=f'D(ℤ_{N})',
            has_complex_topological_data=N > 2,
            num_sectors=N**2,
            descriptive_name=descriptive_name,
        )

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (2,) and np.all(0 <= a) and np.all(a < self.N)

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 2 and np.all(0 <= sectors) and np.all(sectors < self.N)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self.fusion_outcomes_broadcast(a[np.newaxis, :], b[np.newaxis, :])

    def fusion_outcomes_broadcast(self, a: SectorArray, b: SectorArray) -> SectorArray:
        return (a + b) % self.N

    def _multiple_fusion_broadcast(self, *sectors: SectorArray) -> SectorArray:
        return sum(sectors) % self.N

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'QuantumDoubleZNAnyonCategory({self.N}, {name_str})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, QuantumDoubleZNAnyonCategory) and other.N == self.N

    def dual_sector(self, a: Sector) -> Sector:
        return (-a) % self.N

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return (-sectors) % self.N

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return self._phase ** (a[0:1] * b[1:2])

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._phase ** (b[0] * c[1]) * one_4D

    def all_sectors(self) -> SectorArray:
        x = np.arange(self.N, dtype=int)
        return np.dstack(np.meshgrid(x, x)).reshape(-1, 2)


class ToricCodeCategory(QuantumDoubleZNAnyonCategory):
    """Toric code anyon category. Essentially equivalent to `QuantumDoubleZNAnyonCategory(N=2)`.

    The allowed sectors are 1D arrays with two integers between `0` and `1`,
    `[0, 0]`, `[0, 1]`, `[1, 0]`, `[1, 1]`, which are known as vacuum, electric charge,
    magnetic flux and fermion, respectively.

    The electric charges and magnetic fluxes are mutual semions and self-bosons.
    """

    vacuum = as_immutable_array(np.array([0, 0], dtype=int))
    electric_charge = as_immutable_array(np.array([0, 1], dtype=int))
    magnetic_flux = as_immutable_array(np.array([1, 0], dtype=int))
    fermion = as_immutable_array(np.array([1, 1], dtype=int))

    def __init__(self, descriptive_name: str | None = None):
        super().__init__(2, descriptive_name)

    def __repr__(self):
        name_str = '' if self.descriptive_name is None else f'"{self.descriptive_name}"'
        return f'ToricCodeCategory({name_str})'


class FibonacciAnyonCategory(SymmetryFactor):
    """Category describing Fibonacci anyons.

    Allowed sectors are 1D arrays with a single entry of either `0` ("vacuum") or `1` ("tau anyon").
    `[0]`, `[1]`

    `handedness`: ``'left' | 'right'``
        Specifies the chirality / handedness of the anyons. Changing the handedness corresponds to
        complex conjugating the R-symbols, which also affects, e.g., the braid-symbols.
        Considering anyons of different handedness is necessary for doubled models like,
        e.g., the anyons realized in the Levin-Wen string-net models.
    """

    _fusion_map = {  # key: number of tau in fusion input
        0: as_immutable_array(np.array([[0]])),  # 1 x 1 = 1
        1: as_immutable_array(np.array([[1]])),  # 1 x t = t = t x 1
        2: as_immutable_array(np.array([[0], [1]])),  # t x t = 1 + t
    }
    _phi = 0.5 * (1 + np.sqrt(5))  # the golden ratio
    # nontrivial F-symbols
    _f = as_immutable_array(np.expand_dims([_phi**-1, _phi**-0.5, -(_phi**-1)], axis=(1, 2, 3, 4)))
    # nontrivial R-symbols
    _r = as_immutable_array(np.expand_dims([np.exp(-4j * np.pi / 5), np.exp(3j * np.pi / 5)], axis=1))
    vacuum = as_immutable_array(np.array([0], dtype=int))
    tau = as_immutable_array(np.array([1], dtype=int))

    def __init__(self, handedness: Literal['left', 'right'] = 'left'):
        assert handedness in ['left', 'right']
        self.handedness = handedness
        if handedness == 'right':
            self._r = self._r.conj()
        # C++ SymmetryFactor base must be constructed before calling virtual _c_symbol.
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.multiple_unique,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0], dtype=int),
            group_name='FibonacciAnyonCategory',
            has_complex_topological_data=True,
            num_sectors=2,
            descriptive_name=None,
        )
        self._c = [
            _default_c_symbol(self, [0], [1], [1], [0], [1], [1]),
            0,
            0,  # nontrivial C-symbols
            _default_c_symbol(self, [0], [1], [1], [1], [1], [1]),
            0,
            0,
            _default_c_symbol(self, [1], [1], [1], [0], [1], [1]),
            _default_c_symbol(self, [1], [1], [1], [1], [0], [0]),
            _default_c_symbol(self, [1], [1], [1], [1], [1], [0]),
            _default_c_symbol(self, [1], [1], [1], [1], [1], [1]),
        ]

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 2

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 2)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0] + b[0]]

    def sector_str(self, a: Sector) -> str:
        return 'vacuum' if a[0] == 0 else 'tau'

    def __repr__(self):
        return f'FibonacciAnyonCategory(handedness={self.handedness})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, FibonacciAnyonCategory) and other.handedness == self.handedness

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b, c, d])):
            return self._f[e[0] + f[0]]
        return one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1 if a[0] == 0 else self._phi

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.where(a == 1, self._phi, 1).flatten()

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b])):
            return self._r[c[0], :]
        return one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([b, c])):
            return self._c[6 * a[0] + 3 * d[0] + e[0] + f[0] - 2]
        return one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(2, dtype=int)[:, None]


class IsingAnyonCategory(SymmetryFactor):
    """Category describing Ising anyons.

    Allowed sectors are 1D arrays with a single entry of either `0` ("vacuum"), `1` ("Ising anyon")
    or `2` ("fermion").
    `[0]`, `[1]`, `[2]`

    `nu`: odd `int`
        In total, there are 8 distinct Ising models, i.e., `nu` and `nu + 16` describe the same
        anyon model. Different `nu` correspond to different topological twists of the Ising anyons.
        The Ising anyon model of opposite handedness is obtained for `-nu`.
    """

    _fusion_map = {  # 1: vacuum, σ: Ising anyon, ψ: fermion
        0: as_immutable_array(np.array([[0]])),  # 1 x 1 = 1
        1: as_immutable_array(np.array([[1]])),  # 1 x σ = σ = σ x 1
        2: as_immutable_array(np.array([[0], [2]])),  # σ x σ = 1 + ψ
        4: as_immutable_array(np.array([[2]])),  # 1 x ψ = ψ = 1 x ψ
        5: as_immutable_array(np.array([[1]])),  # σ x ψ = σ = σ x ψ
        8: as_immutable_array(np.array([[0]])),  # ψ x ψ = 1
    }
    vacuum = as_immutable_array(np.array([0], dtype=int))
    sigma = as_immutable_array(np.array([1], dtype=int))
    psi = as_immutable_array(np.array([2], dtype=int))

    def __init__(self, nu: int = 1):
        assert nu % 2 == 1
        self.nu = nu % 16
        self.frobenius = as_immutable_array([1, int((-1) ** ((self.nu**2 - 1) / 8)), 1])
        # nontrivial F-symbols
        self._f = as_immutable_array(
            np.expand_dims([1, 0, 1, 0, -1], axis=(1, 2, 3, 4)) * self.frobenius[1] / np.sqrt(2)
        )
        # nontrivial R-symbols
        self._r = as_immutable_array(
            np.expand_dims(
                [
                    (-1j) ** self.nu,
                    -1,
                    np.exp(3j * self.nu * np.pi / 8) * self.frobenius[1],
                    np.exp(-1j * self.nu * np.pi / 8) * self.frobenius[1],
                    0,
                ],
                axis=1,
            )
        )
        # C++ SymmetryFactor base must be constructed before calling virtual _c_symbol.
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.multiple_unique,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0], dtype=int),
            group_name='IsingAnyonCategory',
            has_complex_topological_data=True,
            num_sectors=3,
            descriptive_name=None,
        )
        self._c = [
            (-1j) ** self.nu * one_4D,
            -1 * (-1j) ** self.nu * one_4D,
            _default_c_symbol(self, [0], [1], [1], [0], [1], [1]),  # nontrivial C-symbols
            _default_c_symbol(self, [0], [1], [1], [2], [1], [1]),
            _default_c_symbol(self, [1], [1], [1], [1], [0], [0]),
            _default_c_symbol(self, [1], [1], [1], [1], [0], [2]),
            _default_c_symbol(self, [1], [1], [1], [1], [2], [2]),
            0,
            _default_c_symbol(self, [2], [1], [1], [0], [1], [1]),
            _default_c_symbol(self, [2], [1], [1], [2], [1], [1]),
            -1 * one_4D,
        ]

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 3

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 3)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0] ** 2 + b[0] ** 2]

    def sector_str(self, a: Sector) -> str:
        if a[0] == 1:
            return 'sigma'
        return 'vacuum' if a[0] == 0 else 'psi'

    def __repr__(self):
        return f'IsingAnyonCategory(nu={self.nu})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, IsingAnyonCategory) and other.nu == self.nu

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if not np.any(np.concatenate([a, b, c, d]) - [1, 1, 1, 1]):
            return self._f[e[0] + f[0]]
        elif not np.any(np.concatenate([a, b, c, d]) - [2, 1, 2, 1]):
            return -1 * one_4D
        elif not np.any(np.concatenate([a, b, c, d]) - [1, 2, 1, 2]):
            return -1 * one_4D
        return one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return self.frobenius[a[0]]

    def qdim(self, a: Sector) -> float:
        return np.sqrt(2) if a[0] == 1 else 1

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.where(a == 1, np.sqrt(2), 1).flatten()

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b])):
            return self._r[(a[0] + b[0]) * (c[0] - 1), :]
        return one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if np.all(np.concatenate([b, c])):
            factor = -1 * (b[0] - c[0] - 1) * (b[0] - c[0] + 1)  # = 0 if σ and ψ or σ and ψ, 1 otherwise
            factor *= (
                1 - a[0] // 2 - d[0] // 2 + 9 * (b[0] - 1) + (2 - b[0]) * ((e[0] + f[0]) // 2 + d[0] // 2 + 3 * a[0])
            )
            return self._c[factor + a[0] // 2 + d[0] // 2]
        return one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(3, dtype=int)[:, None]


class SU2_kAnyonCategory(SymmetryFactor):
    """:math:`SU(2)_k` anyon category.

    The anyons can be associated with the spins `0`, `1/2`, `1`, ..., `k/2`.
    Unlike regular SU(2), there is a cutoff at `k/2`.

    Allowed sectors are 1D arrays ``[jj]`` of positive integers `jj` = `0`, `1`, `2`, ..., `k`
    corresponding to `jj/2` listed above.

    Parameters
    ----------
    k : int
        The "level" of the category. ``k/2`` is the largest spin.
    handedness: ``'left' | 'right'``
        Specifies the chirality / handedness of the anyons. Changing the handedness corresponds to
        complex conjugating the R-symbols, which also affects, e.g., the braid-symbols.
        Considering anyons of different handedness is necessary for doubled models like,
        e.g., the anyons realized in the Levin-Wen string-net models.

    """

    # OPTIMIZE : We should introduce caching for the R, F symbols etc.
    #            Probably a simple LRU cache will improve things substantially.
    #            It is unclear if we need to pre-compute, like for SU(N), or if thats overkill

    spin_zero = as_immutable_array(np.array([0], dtype=int))
    spin_half = as_immutable_array(np.array([1], dtype=int))

    def __init__(self, k: int, handedness: Literal['left', 'right'] = 'left'):
        assert isinstance(k, int)
        assert k >= 1
        assert handedness in ['left', 'right']
        self.k = k
        if k >= 2:
            self.spin_one = as_immutable_array(np.array([2], dtype=int))
        self.handedness = handedness
        self._q = np.exp(2j * np.pi / (k + 2))

        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.multiple_unique,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0], dtype=int),
            group_name='SU2_kAnyonCategory',
            num_sectors=self.k + 1,
            has_complex_topological_data=True,
            descriptive_name=None,
        )

        self._r = {}
        for jj1, jj2, jj in product(range(self.k + 1), repeat=3):
            if jj > jj1 + jj2 or jj < abs(jj1 - jj2) or jj1 * jj2 == 0 or jj1 < jj2:
                continue  # do not save trivial R-symbols and use symmetry jj1 <-> jj2
            factor = (-1) ** ((jj - jj1 - jj2) / 2)
            factor *= self._q ** ((jj * (jj + 2) - jj1 * (jj1 + 2) - jj2 * (jj2 + 2)) / 8)
            if self.handedness == 'right':
                factor = factor.conj()
            self._r[(jj1, jj2, jj)] = factor * one_1D

        self._6j = {}
        for jj1, jj2, jj3, jj, jj12, jj23 in product(range(self.k + 1), repeat=6):
            if not (jj1 == np.max([jj1, jj2, jj3, jj, jj12, jj23]) and jj2 == np.max([jj2, jj, jj12, jj23])):
                continue
            jsymbol = self._j_symbol(jj1, jj2, jj12, jj3, jj, jj23)
            if jsymbol != 0:
                self._6j[(jj1, jj2, jj12, jj3, jj, jj23)] = jsymbol

    def _n_q(self, n: int) -> float:
        return (self._q ** (0.5 * n) - self._q ** (-0.5 * n)) / (self._q**0.5 - self._q**-0.5)

    def _n_q_fac(self, n: int) -> float:
        fac = 1
        for i in range(n):
            fac *= self._n_q(i + 1)
        return fac

    def _delta(self, jj1: int, jj2: int, jj3: int) -> float:
        res = self._n_q_fac(round(-1 * jj1 / 2 + jj2 / 2 + jj3 / 2)) * self._n_q_fac(round(jj1 / 2 - jj2 / 2 + jj3 / 2))
        res *= self._n_q_fac(round(jj1 / 2 + jj2 / 2 - jj3 / 2)) / self._n_q_fac(round(jj1 / 2 + jj2 / 2 + jj3 / 2 + 1))
        return np.sqrt(res)

    def _j_symbol(self, jj1: int, jj2: int, jj12: int, jj3: int, jj: int, jj23: int) -> float:
        for triad in [[jj1, jj2, jj12], [jj1, jj, jj23], [jj3, jj2, jj23], [jj3, jj, jj12]]:
            if triad[0] > triad[1] + triad[2] or triad[0] < abs(triad[1] - triad[2]):
                return 0
        start = max([jj1 + jj2 + jj12, jj12 + jj3 + jj, jj2 + jj3 + jj23, jj1 + jj23 + jj]) // 2
        stop = min([jj1 + jj2 + jj3 + jj, jj1 + jj12 + jj3 + jj23, jj2 + jj12 + jj + jj23]) // 2
        res = 0
        for z in range(start, stop + 1):  # runs over all integers for which the factorials have non-negative arguments
            factor = np.prod(
                [
                    self._n_q_fac(round(z - jj1 / 2 - jj2 / 2 - jj12 / 2)),
                    self._n_q_fac(round(z - jj12 / 2 - jj3 / 2 - jj / 2)),
                    self._n_q_fac(round(z - jj2 / 2 - jj3 / 2 - jj23 / 2)),
                    self._n_q_fac(round(z - jj1 / 2 - jj23 / 2 - jj / 2)),
                    self._n_q_fac(round(jj1 / 2 + jj2 / 2 + jj3 / 2 + jj / 2 - z)),
                    self._n_q_fac(round(jj1 / 2 + jj12 / 2 + jj3 / 2 + jj23 / 2 - z)),
                    self._n_q_fac(round(jj2 / 2 + jj12 / 2 + jj / 2 + jj23 / 2 - z)),
                ]
            )
            res += (-1) ** z * self._n_q_fac(z + 1) / factor
        return res * (
            self._delta(jj1, jj2, jj12)
            * self._delta(jj12, jj3, jj)
            * self._delta(jj2, jj3, jj23)
            * self._delta(jj1, jj23, jj)
        )

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a <= self.k

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors <= self.k)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        upper_limit = min(a[0] + b[0], 2 * self.k - a[0] - b[0])
        return np.arange(abs(a[0] - b[0]), upper_limit + 2, 2)[:, np.newaxis]

    def sector_str(self, a: Sector) -> str:
        jj = a[0]
        j_str = str(jj // 2) if jj % 2 == 0 else f'{jj}/2'
        return f'{jj} (j={j_str})'

    def __repr__(self):
        return f'SU2_kAnyonCategory({self.k}, {self.handedness})'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, SU2_kAnyonCategory) and other.k == self.k and other.handedness == self.handedness

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return sectors

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        # The q-deformed 6j symbols have the same symmetries as the usual SU(2) 6j symbols.
        # We can get all f symbols from the cases 6j symbols for
        # a == np.max([a, b, c, d, e, f]) and b == np.max([b, c, e, f]).
        # I.e., we need to exchange the charges accordingly

        # need to compute before exchanging charges
        factor = np.sqrt(self._n_q(e[0] + 1) * self._n_q(f[0] + 1))
        factor *= (-1) ** ((a[0] + b[0] + c[0] + d[0]) / 2)

        argm = np.argmax([a, c, b, d, f, e])
        if argm > 1:
            if argm // 2 == 1:
                a, c, b, d = b, d, a, c
            else:
                a, c, f, e = f, e, a, c

        argm_ = np.argmax([b, d, f, e])
        if argm_ > 1:
            b, d, f, e = f, e, b, d

        if argm % 2 == 1 and argm_ % 2 == 1:
            a, c, b, d = c, a, d, b
        elif argm % 2 == 1:
            a, c, f, e = c, a, e, f
        elif argm_ % 2 == 1:
            b, d, f, e = d, b, e, f

        try:  # nontrivial F-symbols
            return factor * self._6j[(a[0], b[0], f[0], c[0], d[0], e[0])] * one_4D
        except KeyError:
            return one_4D

    def frobenius_schur(self, a: Sector) -> int:
        return -1 if a[0] % 2 == 1 else 1

    def qdim(self, a: Sector) -> float:
        return np.sin((a[0] + 1) * np.pi / (self.k + 2)) / np.sin(np.pi / (self.k + 2))

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.sin((a.flatten() + 1) * np.pi / (self.k + 2)) / np.sin(np.pi / (self.k + 2))

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if a[0] < b[0]:
            a, b = b, a
        try:  # nontrivial R-symbols
            return self._r[(a[0], b[0], c[0])]
        except KeyError:
            return one_1D

    def all_sectors(self) -> SectorArray:
        return np.arange(self.k + 1, dtype=int)[:, None]


class SU3_3AnyonCategory(SymmetryFactor):
    r""":math:`SU(3)_3` anyon category

    Can be used as a good first check for categories with higher fusion multiplicities.

    The anyons are denoted by `1`, `8`, `10` and `\bar{10}` with the fusion rule
    `8 x 8 = 1 + 8 + 8 + 10 + 10-`. (For convenience, we denote `\bar{10}` as `10-`)
    The anyons correspond to the allowed sectors (1D arrays) ``[j]`` with `j = 0,1,2,3`.

    The notion of handedness does not make sense for this specific anyon model since it
    only exchanges the two fusion multiplicities of anyon `8`.
    """

    one_irrep = as_immutable_array([0])
    eight_irrep = as_immutable_array([1])
    ten_irrep = as_immutable_array([2])
    ten_bar_irrep = as_immutable_array([3])

    _fusion_map = {  # notation: 10- = \bar{10}
        0: as_immutable_array([[0]]),  # 1 x 1 = 1
        1: as_immutable_array([[1]]),  # 1 x 8 = 8 = 8 x 1
        4: as_immutable_array([[2]]),  # 1 x 10 = 10 = 1 x 10
        9: as_immutable_array([[3]]),  # 1 x 10- = 10- = 1 x 10-
        2: as_immutable_array([[0], [1], [2], [3]]),  # 8 x 8 = 1 + 8 + 8 + 10 + 10-
        5: as_immutable_array([[1]]),  # 8 x 10 = 8 = 10 x 8
        10: as_immutable_array([[1]]),  # 8 x 10- = 8 = 10- x 8
        8: as_immutable_array([[3]]),  # 10 x 10 = 10-
        13: as_immutable_array([[0]]),  # 10 x 10- = 1 = 10- x 10
        18: as_immutable_array([[2]]),  # 10- x 10- = 10
    }
    _dual_map = {
        0: as_immutable_array([0]),
        1: as_immutable_array([1]),
        2: as_immutable_array([3]),
        3: as_immutable_array([2]),
    }
    _f1 = as_immutable_array(np.identity(2))
    _f2 = as_immutable_array([[-0.5, -(3**0.5) / 2], [3**0.5 / 2, -0.5]])
    _f3 = _f2.T
    _f4 = np.zeros((7, 7))
    _f4[0, 0] = _f4[5, 5] = _f4[6, 5] = _f4[5, 6] = _f4[6, 6] = 1 / 3
    _f4[0, 5] = _f4[0, 6] = _f4[5, 0] = _f4[6, 0] = -1 / 3
    _f4[0, 1] = _f4[1, 0] = _f4[0, 4] = _f4[4, 0] = 3**-0.5
    _f4[2, 2] = _f4[3, 2] = _f4[2, 3] = _f4[3, 3] = _f4[1, 4] = _f4[4, 1] = 0.5
    _f4[2, 6] = _f4[6, 3] = _f4[3, 5] = _f4[5, 2] = 0.5
    _f4[2, 5] = _f4[5, 3] = _f4[3, 6] = _f4[6, 2] = -0.5
    _f4[1, 1] = _f4[4, 4] = -0.5
    _f4[1, 5] = _f4[1, 6] = _f4[5, 1] = _f4[6, 1] = 12**-0.5
    _f4[4, 5] = _f4[4, 6] = _f4[5, 4] = _f4[6, 4] = 12**-0.5
    _f4 = as_immutable_array(_f4)
    _fsym_map = {}

    def __init__(self):
        self._c = {}
        SymmetryFactor.__init__(
            self,
            fusion_style=FusionStyle.general,
            braiding_style=BraidingStyle.anyonic,
            trivial_sector=np.array([0], dtype=int),
            group_name='SU3_3AnyonCategory',
            num_sectors=4,
            has_complex_topological_data=True,
            descriptive_name=None,
        )

        for charges in product(range(4), repeat=6):
            a, b, c, d, e, f = [np.array([i]) for i in charges]
            self._fsym_map[(a[0], b[0], c[0], d[0], e[0], f[0])] = self._compute_f_symbol(a, b, c, d, e, f)

        for charges in product(range(4), repeat=6):
            a, b, c, d, e, f = [np.array([i]) for i in charges]
            if (
                self.can_fuse_to(a, b, e)
                and self.can_fuse_to(e, c, d)
                and self.can_fuse_to(a, c, f)
                and self.can_fuse_to(f, b, d)
            ):
                self._c[(a[0], b[0], c[0], d[0], e[0], f[0])] = _default_c_symbol(self, a, b, c, d, e, f)

    def _compute_f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        if not np.all(
            [self.can_fuse_to(b, c, e), self.can_fuse_to(a, e, d), self.can_fuse_to(a, b, f), self.can_fuse_to(f, c, d)]
        ):
            return one_4D

        abcd = [a, b, c, d]
        check_8 = [charge == np.array([1]) for charge in abcd]
        shape = (self._n_symbol(b, c, e), self._n_symbol(a, e, d), self._n_symbol(a, b, f), self._n_symbol(f, c, d))

        if check_8.count(True) == 4:
            slices = []
            for charge in [e, f]:
                if charge == np.array([0]):
                    slices.append(slice(0, 1))
                elif charge == np.array([1]):
                    slices.append(slice(1, 5))
                elif charge == np.array([2]):
                    slices.append(slice(5, 6))
                else:
                    slices.append(slice(6, 7))
            return self._f4[slices[1], slices[0]].reshape(shape)

        elif check_8.count(True) == 3:
            index = check_8.index(False)
            not_8 = abcd[index]
            if not_8 == self.trivial_sector:
                return self._f1.reshape(shape)
            elif (not_8 == np.array([2]) and index != 1) or (not_8 == np.array([3]) and index == 1):
                return self._f2.reshape(shape)
            else:
                return self._f3.reshape(shape)

        elif check_8.count(True) == 2 and np.all(abcd):  # two 8 and no 1
            index1 = check_8.index(True)
            check_8[index1] = False
            index2 = check_8.index(True)
            if (index2 == index1 + 1) or (index1 == 0 and index2 == 3):
                return -1 * one_4D

        elif check_8.count(True) == 0 and np.all(abcd):
            check_10 = [charge == np.array([2]) for charge in abcd]
            index = 1
            if check_10.count(True) == 3:
                index = check_10.index(False)
            elif check_10.count(True) == 1:
                index = check_10.index(True)
            if index == 0 or index == 2:
                return -1 * one_4D
        return one_4D

    def is_valid_sector(self, a: Sector) -> bool:
        return getattr(a, 'shape', ()) == (1,) and 0 <= a < 4

    def are_valid_sectors(self, sectors) -> bool:
        shape = getattr(sectors, 'shape', ())
        return len(shape) == 2 and shape[1] == 1 and np.all(0 <= sectors) and np.all(sectors < 4)

    def fusion_outcomes(self, a: Sector, b: Sector) -> SectorArray:
        return self._fusion_map[a[0] ** 2 + b[0] ** 2]

    def sector_dim(self, a: Sector) -> int:
        return 1

    def batch_sector_dim(self, a: SectorArray) -> np.ndarray:
        return np.ones((len(a),), int)

    def sector_str(self, a: Sector) -> str:
        if a[0] == 1:
            return 'eight'
        elif a[0] == 2:
            return 'ten'
        return 'one' if a[0] == 0 else 'ten_bar'

    def __repr__(self):
        return f'SU3_3AnyonCategory()'

    def _is_equivalent_factor(self, other) -> bool:
        return isinstance(other, SU3_3AnyonCategory)

    def dual_sector(self, a: Sector) -> Sector:
        return self._dual_map[a[0]]

    def dual_sectors(self, sectors: SectorArray) -> SectorArray:
        return np.where(sectors >= 2, -sectors % 5, sectors)

    def _n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 2 if np.all(np.concatenate([a, b, c]) == np.array([[1] * 3])) else 1

    def _f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        return self._fsym_map[(a[0], b[0], c[0], d[0], e[0], f[0])]

    def frobenius_schur(self, a: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 3 if a[0] == 1 else 1

    def batch_qdim(self, a: SectorArray) -> np.ndarray:
        return np.where(a == 1, 3, 1).flatten()

    def _r_symbol(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        if np.all(np.concatenate([a, b]) == np.array([[1], [1]])):
            if c == np.array([1]):
                return np.array([-1j, 1j])
            return -1 * one_1D
        return one_1D

    def _c_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> np.ndarray:
        try:
            return self._c[(a[0], b[0], c[0], d[0], e[0], f[0])]
        except KeyError:  # inconsistent fusion
            return one_4D

    def all_sectors(self) -> SectorArray:
        return np.arange(4, dtype=int)[:, None]


# Note : some symmetries have expensive __init__ ! Do not initialize those.
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
