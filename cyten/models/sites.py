"""Defines classes that describe the sites of a lattice."""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np
from typing import Literal

from ..spaces import ElementarySpace
from ..symmetries import (
    SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory, SU2_kAnyonCategory
)
from .degrees_of_freedom import (
    SpinDOF, FermionicDOF, BosonicDOF, ClockDOF, RepresentationDOF
)


class SpinSite(SpinDOF):
    """TODO elaborate"""

    def __init__(self, S: float = .5, conserve: Literal['SU(2)', 'Sz', 'parity', 'None'] = None):
        self.S = S = float(S)
        two_S = int(round(2 * S, 0))
        dim = two_S + 1
        if two_S < 0:
            raise ValueError('Negative spin.')
        if not np.allclose(two_S / 2, S):
            raise ValueError('total_spin must be half integer: 0, 1/2, 1, 3/2, ...')

        # build spin vector
        Sz = np.diag(-S + np.arange(dim))
        Sp = np.zeros((dim, dim))
        for n in range(dim - 1):
            # Sp |m> = sqrt( S(S+1) - m(m+1) ) |m+1>
            m = n - S
            Sp[n + 1, n] = np.sqrt((S * (S + 1) - m * (m + 1)))
        spin_vector = self._spin_vector_from_Sp(Sz=Sz, Sp=Sp)

        # build leg
        if conserve in ['SU(2)', 'SU2']:
            sym = SU2Symmetry('spin')
            leg = ElementarySpace.from_defining_sectors(sym, [[two_S]])
        elif conserve in ['Sz', 'U(1)', 'U1']:
            sym = U1Symmetry('2*Sz')
            leg = ElementarySpace.from_basis(sym, np.arange(-two_S, two_S + 2, 2)[:, None])
        elif conserve in ['parity', 'Z_2', 'Z2']:
            sym = ZNSymmetry(2, 'Sz_parity')
            leg = ElementarySpace.from_basis(sym, np.arange(dim)[:, None] % 2)
        elif conserve in ['None', 'none', None]:
            sym = NoSymmetry()
            leg = ElementarySpace.from_trivial_sector(dim=dim, symmetry=sym)
        else:
            raise ValueError(f'Invalid `conserve`: {conserve}')
        self.conserve = conserve

        state_labels = {str(n - S): n for n in range(dim)}
        state_labels['down'] = 0
        state_labels['up'] = dim - 1

        SpinDOF.__init__(self, leg=leg, double_total_spin=two_S, spin_vector=spin_vector,
                         spin_symmetry=sym, spin_symmetry_sector_slice=slice(None, None),
                         state_labels=state_labels)

    def __repr__(self):
        return f'SpinSite(S={self.S}, conserve={self.conserve})'


class SpinlessBosonSite(BosonicDOF):
    """TODO elaborate"""


class SpinlessFermionSite(FermionicDOF):
    """TODO elaborate"""


class SpinHalfFermionSite(SpinDOF, FermionicDOF):
    """TODO similar to SpinSite..."""


class ClockSite(ClockDOF):
    """TODO elaborate"""

    def __init__(self, q: int, conserve: Literal['Z_N', 'None'] = None):
        assert isinstance(q, int)

        # build clock operators
        X = np.eye(q, k=1) + np.eye(q, k=1 - q)
        Z = np.diag(np.exp(2.j * np.pi * np.arange(q, dtype=np.complex128) / q))
        clock_operators = np.stack([X, Z], axis=2)

        # build leg
        if conserve in ['Z_N', 'ZN', 'Z_q', 'Zq']:
            sym = ZNSymmetry(q, 'q')
            leg = ElementarySpace.from_basis(sym, np.arange(q)[:, None])
        elif conserve in ['None', 'none', None]:
            sym = NoSymmetry()
            leg = ElementarySpace.from_trivial_sector(dim=q, symmetry=sym)
        else:
            raise ValueError(f'Invalid `conserve`: {conserve}')
        self.conserve = conserve

        state_labels = {str(n): n for n in range(q)}
        state_labels['up'] = 0
        if q % 2 == 0:
            state_labels['down'] = q // 2

        ClockDOF.__init__(self, leg=leg, q=q, clock_operators=clock_operators, clock_symmetry=sym,
                          clock_symmetry_sector_slice=slice(None, None), state_labels=state_labels)

    def __repr__(self):
        return f'ClockSite(q={self.q}, conserve={self.conserve})'


class GoldenSite(RepresentationDOF):
    """Base class for the golden chain model where the local Hilbert space is the tau sector"""

    def __init__(self, handedness: Literal['left', 'right']):
        symmetry = FibonacciAnyonCategory(handedness=handedness)
        RepresentationDOF.__init__(self, symmetry, [symmetry.tau], [1])


class SU2kSite(RepresentationDOF):
    """Base class for SU2_k sites where each site is a direct sum of simple objects of SU2_k with multiplicities."""

    def __init__(self, k, handedness, simples: list, multiplicities: list):

        symmetry = SU2_kAnyonCategory(k, handedness=handedness)
        RepresentationDOF.__init__(self, symmetry, simples, multiplicities)


# TODO more sites:
#  - FermionSite (maybe name it SpinlessFermionSite for clarity?)
#  - SpinHalfFermionSite (or if its easy just do general spin?)
#  - SpinHalfHoleSite (i dont think this should inherit from FermionicSite, but not sure)
#  - BosonSite (maybe name it SpinlessBosonSite?)
#  - bosons with spin?
#  - what are relevant anyonic sites? already have Golden, but do some more
#  - remember to update cyten/__init__.py and cyten/models/__init__.py accordingly!
