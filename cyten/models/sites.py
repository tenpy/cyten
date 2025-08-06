"""Defines classes that describe the sites of a lattice."""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np
from typing import Literal
from itertools import product as itproduct

from ..spaces import ElementarySpace
from ..symmetries import (
    ProductSymmetry, SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory,
    SU2_kAnyonCategory, FermionParity
)
from .degrees_of_freedom import (
    SpinDOF, FermionicDOF, BosonicDOF, ClockDOF, RepresentationDOF
)


class SpinSite(SpinDOF):
    """Class for sites that have a single spin degree of freedom.

    TODO find a good format to doc the onsite operators that exist in a site

    Attributes
    ----------
    S : float
        The total spin.
    double_total_spin : int
        Twice the :attr:`S`. We store this in addition because it is an integer.
    conserve : Literal['SU(2)', 'Sz', 'parity', 'None']
        The symmetry to be conserved. We can conserve::

            - SU(2), the full spin rotation symmetry.
            - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
            - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
            - nothing.
    """

    def __init__(self, S: float = .5, conserve: Literal['SU(2)', 'Sz', 'parity', 'None'] = None):
        self.S = S = float(S)
        two_S = int(round(2 * S, 0))
        self.double_total_spin = two_S
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

        SpinDOF.__init__(self, leg=leg, spin_vector=spin_vector, state_labels=state_labels)

        if not isinstance(sym, SU2Symmetry):
            self.add_onsite_operator('Sz', spin_vector[:, :, 2])
        if isinstance(sym, NoSymmetry):
            self.add_onsite_operator('Sx', spin_vector[:, :, 0])
            self.add_onsite_operator('Sy', spin_vector[:, :, 1])

    def test_sanity(self):
        super().test_sanity()
        S_sq = np.tensordot(self.spin_vector, self.spin_vector, ([-1, 1], [-1, 0]))
        eigenvalue = self.double_total_spin * (self.double_total_spin + 2) / 4
        assert np.allclose(S_sq, eigenvalue * np.eye(self.double_total_spin + 1))

    def __repr__(self):
        return f'SpinSite(S={self.S}, conserve={self.conserve})'


class SpinlessBosonSite(BosonicDOF):
    """Site for (possibly multiple) spinless bosons.

    TODO describe onsite operators

    Parameters
    ----------
    Nmax : int | list[int] | np.ndarray[int]
        The maximum occupation of each of the boson species. An `int` corresponds to a single boson
        species. Otherwise, the number of boson species corresponds to `len(Nmax)`.
    conserve : Literal['N', 'parity', 'None'] | list[Literal['N', 'parity', 'None']]
        The symmetry to be conserved. We can conserve::

            - total particle number sum_i N_i (``conserve == 'N'``).
            - individual particle numbers N_i (``conserve[i] == 'N'``).
            - total parity (sum_i N_i) % 2 (``conserve == 'parity'``).
            - individual parities N_i % 2 (``conserve[i] == 'parity'``).
            - nothing (``conserve == 'None'`` or ``conserve[i] == 'None'``).

        A `Literal` corresponds to symmetries involving all boson species, such as the total
        particle number (``conserve == 'N'``) or the total parity (``conserve == 'parity'``).
        For a list, the entry ``conserve[i]`` corresponds to the symmetry of boson species `i`,
        such that, e.g., ``conserve[i] == 'N'`` signifies that its particle number is conserved.

        Conserves nothing by default.
    filling : float | None | list[float | None] | np.ndarray[float | None]
        Average filling for each species. Used to define the on-site operators ``dN`` and ``dNdN``
        if ``filling is not None`` or ``filling[i] is not None``. A `float` or `None` is by default
        applied to every boson species

    Attributes
    ----------
    conserve : Literal['N', 'parity', 'None'] | list[Literal['N', 'parity', 'None']]
        The conserved symmetry, see above.
    filling : np.ndarray[float | None]
        Average filling for each species.
    num_species, Nmax, creators, annihilators : see :class:`BosonicDOF`
    """

    def __init__(self, Nmax: int | list[int] | np.ndarray[int],
                 conserve: Literal['N', 'parity', 'None'] | list[Literal['N', 'parity', 'None']] = None,
                 filling: float | None | list[float | None] | np.ndarray[float | None] = None):
        Nmax = np.atleast_1d(np.asarray(Nmax, dtype=int))
        # need to manually throw an error for non-integers in Nmax
        assert np.allclose(Nmax, np.asarray(Nmax)), f'Invalid `Nmax`: {Nmax}'
        num_species = len(Nmax)
        if isinstance(conserve, (list, np.ndarray)):
            msg = f'Invalid number of entries in `conserve`: {len(conserve)} != {num_species}'
            assert len(conserve) == num_species, msg
        if isinstance(filling, (list, np.ndarray)):
            msg = f'Invalid number of entries in `filling`: {len(filling)} != {num_species}'
            assert len(filling) == num_species, msg
        else:
            filling = [filling] * num_species
        self.filling = np.asarray(filling)

        # states for each species
        states = [list(range(n + 1)) for n in Nmax]
        dims = np.ones_like(Nmax) + Nmax
        total_dim = np.prod(dims, dtype=int)

        if isinstance(conserve, (list, np.ndarray)):
            sym_factors = []
            no_sym_idcs = []
            parity_sym_idcs = []
            for i, conserve_i in enumerate(conserve):
                if conserve_i in ['N', 'Ni', 'N_i', 'U(1)', 'U1']:
                    sym_factors.append(U1Symmetry(f'species{i}_occupation'))
                elif conserve_i in ['parity', 'P', 'Pi', 'P_i', 'Z_2', 'Z2']:
                    sym_factors.append(ZNSymmetry(2, f'species{i}_occupation_parity'))
                    parity_sym_idcs.append(i)
                elif conserve_i in ['None', 'none', None]:
                    sym_factors.append(NoSymmetry())
                    no_sym_idcs.append(i)
                else:
                    raise ValueError(f'Invalid entry in `conserve`: {conserve_i}')

            if len(no_sym_idcs) == num_species:
                sym = NoSymmetry()
                leg = ElementarySpace.from_trivial_sector(dim=total_dim, symmetry=sym)
            else:
                sym = ProductSymmetry(sym_factors)
                sectors = []
                for occupations in itproduct(*states):
                    sector = np.asarray(occupations, dtype=int)
                    sector[no_sym_idcs] = 0
                    sector[parity_sym_idcs] = np.mod(sector[parity_sym_idcs], 2)
                    sectors.append(sector)
                leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int))
        else:
            # for U(1) and Z_2, iterate over all states in the correct order to
            # get the correct basis_perm in ElementarySpace.from_basis
            if conserve in ['N', 'Ntot', 'N_tot', 'U(1)', 'U1']:
                sym = U1Symmetry('total_occupation')
                sectors = []
                for occupations in itproduct(*states):
                    sectors.append(np.sum(occupations))
                leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int)[:, None])
            elif conserve in ['parity', 'P', 'Ptot', 'P_tot', 'Z_2', 'Z2']:
                sym = ZNSymmetry(2, 'total_occupation_parity')
                sectors = []
                for occupations in itproduct(*states):
                    sectors.append(np.sum(occupations) % 2)
                leg = ElementarySpace.from_basis(sym, np.asarray(sectors, dtype=int)[:, None])
            elif conserve in ['None', 'none', None]:
                sym = NoSymmetry()
                leg = ElementarySpace.from_trivial_sector(dim=total_dim, symmetry=sym)
            else:
                raise ValueError(f'Invalid `conserve`: {conserve}')
        self.conserve = conserve

        # state labels have the form '(n0, n1, ...)' with n0, n1, ... corresponding to the
        # occupations for the species. For a single species, this is changed to 'n0', i.e.,
        # the brackets and comma from the tuple are omitted.
        state_labels = {}
        dim_prod = np.asarray([np.prod(dims[i + 1:]) for i in range(num_species)], dtype=int)
        for occupations in itproduct(*states):
            label = str(occupations)
            if num_species == 1:
                label = label[1:-2]
            state_labels[label] = np.sum(np.asarray(occupations, dtype=int) * dim_prod)
        # vacuum == no bosons
        state_labels['vac'] = 0

        creators, annihilators = BosonicDOF._creation_annihilation_ops_from_Nmax(Nmax=Nmax)

        # construct operators relative to filling for each entry that is not None
        ops = {}
        for i, filling_i in enumerate(self.filling):
            if filling_i is None:
                continue
            N_i_diag = np.diag(creators[:, :, i] @ annihilators[:, :, i])
            dN_i = np.diag(N_i_diag - filling_i * np.ones(total_dim))
            dNdN_i = np.diag((N_i_diag - filling_i * np.ones(total_dim)) ** 2)
            if num_species == 1:
                ops[f'dN'] = dN_i
                ops[f'dNdN'] = dNdN_i
            else:
                ops[f'dN{i}'] = dN_i
                ops[f'dN{i}dN{i}'] = dNdN_i

        BosonicDOF.__init__(self, leg=leg, creators=creators, annihilators=annihilators,
                            state_labels=state_labels, onsite_operators=ops)

    def __repr__(self):
        return f'SpinlessBosonSite(Nmax={self.Nmax}, conserve={self.conserve}, filling={self.filling})'


class SpinlessFermionSite(FermionicDOF):
    """Site for (possibly multiple) spinless fermions.

    TODO describe onsite operators

    .. todo ::
        For now, assume that the symmetry needs to capture the fermionic statistics.
        Do not think about JW strings yet...
        That is also the reason why NoSymmetry is not an option here

    Parameters
    ----------
    num_species : int
        Number of fermion species.
    conserve : Literal['N', 'parity'] | list[Literal['N', 'parity', 'None']]
        The symmetry to be conserved. We can conserve::

            - total fermion number sum_i N_i (``conserve == 'N'``).
            - individual fermion numbers N_i (``conserve[i] == 'N'``).
            - total fermion parity (sum_i N_i) % 2 (``conserve == 'parity'``).
            - individual fermion parities N_i % 2 (``conserve[i] == 'parity'``).
            - nothing for an individual fermion (``conserve[i] == 'None'``); .

        A `Literal` corresponds to symmetries involving all fermion species, such as the total
        fermion number (``conserve == 'N'``) or the total fermion parity
        (``conserve == 'parity'``). For a list, the entry ``conserve[i]`` corresponds to the
        symmetry of fermion species `i`, such that, e.g., ``conserve[i] == 'N'`` signifies that
        its fermion number is conserved.

        Note that the total fermion parity is always conserved. It is thus always part of the
        symmetry. Hence, ``conserve == 'None'`` is not a valid value. On the other hand,
        ``conserve = ['None']`` is interpreted as valid and the resulting symmetry conserves the
        fermionic parity.

        Conserves total fermion parity by default.
    filling : float | None | list[float | None] | np.ndarray[float | None]
        Average filling for each species. Used to define the on-site operators ``dN`` and ``dNdN``
        if ``filling is not None`` or ``filling[i] is not None``. A `float` or `None` is by default
        applied to every fermion species.

    Attributes
    ----------
    num_species : int
        Number of fermion species.
    conserve : Literal['N', 'parity'] | list[Literal['N', 'parity', 'None']]
        The conserved symmetry, see above.
    filling : np.ndarray[float | None]
        Average filling for each species.
    creators, annihilators : see :class:`FermionicDOF`
    """

    def __init__(self, num_species: int,
                 conserve: Literal['N', 'parity'] | list[Literal['N', 'parity', 'None']] = 'parity',
                 filling: float | None | list[float | None] | np.ndarray[float | None] = None):
        assert isinstance(num_species, int)
        assert num_species > 0, 'Must have at least a single fermion species'
        if isinstance(conserve, (list, np.ndarray)):
            msg = f'Invalid number of entries in `conserve`: {len(conserve)} != {num_species}'
            assert len(conserve) == num_species, msg
        if isinstance(filling, (list, np.ndarray)):
            msg = f'Invalid number of entries in `filling`: {len(filling)} != {num_species}'
            assert len(filling) == num_species, msg
        else:
            filling = [filling] * num_species
        self.filling = np.asarray(filling)

        if isinstance(conserve, (list, np.ndarray)):
            sym_factors = []
            no_sym_idcs = []
            parity_sym_idcs = []
            for i, conserve_i in enumerate(conserve):
                if conserve_i in ['N', 'Ni', 'N_i']:
                    sym_factors.append(U1Symmetry(f'species{i}_fermion_occupation'))
                elif conserve_i in ['parity', 'P', 'Pi', 'P_i']:
                    sym_factors.append(ZNSymmetry(2, f'species{i}_fermion_parity'))
                    parity_sym_idcs.append(i)
                elif conserve_i in ['None', 'none', None]:
                    sym_factors.append(NoSymmetry())
                    no_sym_idcs.append(i)
                else:
                    raise ValueError(f'Invalid entry in `conserve`: {conserve_i}')

            if len(no_sym_idcs) == num_species:
                sym = FermionParity('total_fermion_parity')
            else:
                sym = ProductSymmetry([*sym_factors, FermionParity('total_fermion_parity')])
                sectors = []
                for occupations in itproduct([0, 1], repeat=num_species):
                    sector = np.asarray(occupations, dtype=int)
                    sector = np.append(sector, np.sum(sector) % 2)
                    sector[no_sym_idcs] = 0
                    sector[parity_sym_idcs] = np.mod(sector[parity_sym_idcs], 2)
                    sectors.append(sector)
                leg = ElementarySpace.from_defining_sectors(sym, np.asarray(sectors, dtype=int))
        else:
            if conserve in ['N', 'Ntot', 'N_tot']:
                sym = ProductSymmetry([U1Symmetry('total_fermion_occupation'),
                                       FermionParity('total_fermion_parity')])
                sectors = []
                mults = []
                for fermion_number in range(num_species + 1):
                    sectors.append([fermion_number, fermion_number % 2])
                    mults.append(FermionicDOF._states_with_occupation(fermion_number, num_species))
                leg = ElementarySpace.from_defining_sectors(sym, sectors, mults,
                                                            unique_sectors=True)
            elif conserve in ['parity', 'P', 'Ptot', 'P_tot']:
                sym = FermionParity('total_fermion_parity')
            else:
                raise ValueError(f'Invalid `conserve`: {conserve}')

        # conserve == 'parity' and conserve == ['None', ..., 'None'] are the same
        if isinstance(sym, FermionParity):
            sectors = np.asarray([[0], [1]], dtype=int)
            mults = np.asarray([2 ** (num_species - 1)] * 2, dtype=int)
            leg = ElementarySpace(sym, sectors, mults)
        self.conserve = conserve

        creators, annihilators = FermionicDOF._creation_annihilation_ops(num_species=num_species)

        # construct occupation operators, total occupation op, total parity op,
        # and ops relative to filling for each entry that is not None
        # must be done as SymmetricTensor since basis_perm is ill-defined
        ops = {}
        # TODO

        FermionicDOF.__init__(self, leg=leg, creators=creators, annihilators=annihilators,
                              state_labels=None, onsite_operators=ops)


class SpinHalfFermionSite(SpinDOF, FermionicDOF):
    """TODO similar to SpinSite..."""


class ClockSite(ClockDOF):
    """Class for sites that have a single quantum clock degree of freedom.

    TODO describe onsite operators

    Parameters
    ----------
    q : int
        Number of states per site.
    conserve : Literal['Z_N', 'None']
        The symmetry to be conserved. We can conserve::

            - Z_N symmetry.
            - nothing.

    Attributes
    ----------
    conserve : Literal['Z_N', 'None']
        The conserved symmetry, see above.
    q, clock_operators : see :class:`ClockDOF`
    """

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

        ClockDOF.__init__(self, leg=leg, q=q, clock_operators=clock_operators,
                          state_labels=state_labels)

        Xhc = np.conj(clock_operators[:, :, 0].T)
        if isinstance(sym, NoSymmetry):
            self.add_onsite_operator('X', X)
            self.add_onsite_operator('Xhc', Xhc)
            self.add_onsite_operator('Xphc', X + Xhc)

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
