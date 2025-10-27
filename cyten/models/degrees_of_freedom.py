"""Defines classes describing the local physical Hilbert spaces.

The :class:`DegreeOfFreedom` is the prototype, read its docstring.
All other classes are base classes from which sites are derived.
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np
from typing import Literal, Sequence
from functools import reduce
from math import comb
from abc import abstractmethod, ABCMeta

from ..backends import TensorBackend, get_backend
from ..backends.abstract_backend import Block
from ..spaces import ElementarySpace
from ..tensors import DiagonalTensor, SymmetricTensor
from ..symmetries import (
    FermionNumber, FermionParity, U1Symmetry, ZNSymmetry, SU2Symmetry, Sector, Symmetry,
    NoSymmetry, ProductSymmetry, BraidingStyle, SymmetryError
)
from ..tools import to_iterable, is_iterable, to_valid_idx, as_immutable_array


ALL_SPECIES = object()
"""Singleton object used to indicate to sum over all species in fermion/boson couplings."""


class Site:
    """Collects necessary information about a local site of a lattice model.

    A site defines the local Hilbert space in terms of its :attr:`leg`.
    This involves a choice for the local basis.
    Moreover, it exposes the symmetric single-site operators.
    Multi-site operators, on the other hand, are represented by :class:`Coupling` s.

    Attributes
    ----------
    leg : ElementarySpace
        The local physical Hilbert space.
    state_labels : {str: int}
        Optional labels for the local basis states. Any state may have multiple labels, or none.
    onsite_operators : {str: SymmetricTensor}
        The available on-site operators. Note: which operators are available typically depends
        on what symmetry is enforced. Operators that are symmetric under a small symmetry may
        not be symmetric under a larger symmetry, and are thus not available as `onsite_operators`.
        Each must have the :attr:`leg` of the site as the only factor in its domain and codomain.

    Examples
    --------
    TODO
    """

    def __init__(self, leg: ElementarySpace, state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None, default_device: str = None):
        self.leg = leg
        if state_labels is None:
            state_labels = {}
        self.state_labels = state_labels
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        self.backend = backend
        if default_device is None:
            default_device = 'cpu'
        self.default_device = default_device
        self.onsite_operators: dict[str, SymmetricTensor] = {}
        if onsite_operators is not None:
            for name, op in onsite_operators.items():
                # TODO calling it like this does not allow fermionic operators to be constructed
                # -> must be constructed with add_onsite_operator outside the __init__
                self.add_onsite_operator(name, op)

    def test_sanity(self):
        self.leg.test_sanity()

        # state labels
        if self.symmetry.braiding_style >= BraidingStyle.anyonic:
            # can not have state labels, since we dont have basis states in the strict sense
            assert len(self.state_labels) == 0
        for label, idx in self.state_labels.items():
            assert isinstance(label, str)
            assert 0 <= idx < self.dim

        # onsite_operators
        for op in self.onsite_operators.values():
            assert op.codomain.factors == [self.leg] == op.domain.factors
            assert op.labels == ['p', 'p*']
            op.test_sanity()

    @property
    def symmetry(self) -> Symmetry:
        return self.leg.symmetry

    @property
    def dim(self) -> int | float:
        return self.leg.dim

    def add_onsite_operator(self, name: str, op: SymmetricTensor | Block,
                            is_diagonal: bool = None, understood_braiding: bool = False):
        """Add an operator to the :attr:`onsite_operators`."""
        if name in self.onsite_operators:
            raise ValueError(f'Operator with {name=} already exists.')
        #
        if isinstance(op, SymmetricTensor):
            if is_diagonal is not None:
                assert isinstance(op, DiagonalTensor) == bool(is_diagonal)
            assert op.codomain.factors == [self.leg]
            assert op.domain.factors == [self.leg]
            if op.labels != ['p', 'p*']:
                op = op.copy(deep=False)
                op.labels = ['p', 'p*']
        elif is_diagonal is True:
            op = DiagonalTensor.from_dense_block(
                block=op, leg=self.leg, backend=self.backend, labels=['p', 'p*'],
                device=self.default_device, understood_braiding=understood_braiding
            )
        else:
            op = SymmetricTensor.from_dense_block(
                block=op, codomain=[self.leg], domain=[self.leg], backend=self.backend,
                labels=['p', 'p*'], device=self.default_device, understood_braiding=understood_braiding
            )
        self.onsite_operators[name] = op

    def state_index(self, label: str | int) -> int:
        """The index of a basis state."""
        if isinstance(label, str):
            try:
                return self.state_labels[label]
            except KeyError:
                raise KeyError(f'Label not found: {label}') from None
        res = int(label)
        if not -self.dim <= res < self.dim:
            raise ValueError('Index out of bounds')
        if res < 0:
            return res + self.dim
        return res

    def state_indices(self, labels: Sequence[str | int]) -> list[int]:
        """The indices of multiple basis states"""
        return [self.state_index(l) for l in labels]

    def __repr__(self):
        return f'<{type(self).__name__}, dim={self.dim}, symmetry={self.symmetry}>'


class SpinDOF(Site):
    """Common base class for sites that have a spin degree of freedom.

    Attributes
    ----------
    spin_vector : 3D array
        The vector of spin operators as a numpy array with axes ``[p, p*, i]`` and shape
        ``(dim, dim, 3)``. These operators include the factor of the total spin,
        e.g. for spin-1/2, these are ``.5`` times the pauli matrices.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 spin_vector: np.ndarray,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        assert spin_vector.shape == (leg.dim, leg.dim, 3)
        self.spin_vector = as_immutable_array(spin_vector)
        super().__init__(
            leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )

    def test_sanity(self):
        super().test_sanity()
        # check commutation relations
        Sx, Sy, Sz = [self.spin_vector[:, :, i] for i in range(3)]
        assert np.allclose(Sx @ Sy - Sy @ Sx, 1j * Sz)
        assert np.allclose(Sy @ Sz - Sz @ Sy, 1j * Sx)
        assert np.allclose(Sz @ Sx - Sx @ Sz, 1j * Sy)

    @staticmethod
    def conservation_law_to_symmetry(conserve: Literal['SU(2)', 'Sz', 'parity', 'None']
                                     ) -> Symmetry:
        """Translate conservation law for a spin to a symmetry."""
        if conserve in ['SU(2)', 'SU2']:
            sym = SU2Symmetry('spin')
        elif conserve in ['Sz', 'U(1)', 'U1']:
            sym = U1Symmetry('2*Sz')
        elif conserve in ['parity', 'Sz_parity', 'Z_2', 'Z2']:
            sym = ZNSymmetry(2, 'Sz_parity')
        elif conserve in ['None', 'none', None]:
            sym = NoSymmetry()
        else:
            raise ValueError(f'Invalid `conserve`: {conserve}')
        return sym

    @staticmethod
    def _spin_vector_from_Sp(Sz: np.ndarray, Sp: np.ndarray) -> np.ndarray:
        """Build the spin_vector from ``Sz`` and ``Sp = Sx + i Sy``"""
        dim = Sz.shape[0]
        assert Sz.shape == (dim, dim)
        assert Sp.shape == (dim, dim)
        Sm = Sp.T.conj()
        Sx = .5 * (Sp + Sm)
        Sy = .5j * (Sm - Sp)
        return np.stack([Sx, Sy, Sz], axis=-1)


class OccupationDOF(Site, metaclass=ABCMeta):
    """Common base class for sites that have a bosonic or fermionic degree of freedom.

    Requires that the local basis is such that the :attr:`number_operators` of all species
    are diagonal.

    Attributes
    ----------
    num_species : int
        Number of boson species.
    creators : 3D array
        The vector of creation operators as a numpy array with shape ``(dim, dim, num_species)``
        and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
        ``[Bd0, Bd1`, ...]`` stacked along axis 2).
    annihilators : 3D array
        The vector of annihilation operators as a numpy array with shape ``(dim, dim, num_species)``
        and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
        ``[B0, B1`, ...]`` stacked along axis 2).
    anti_commute_sign : float
        ``+1`` for bosons, ``-1`` for fermions.
    species_names : list of (str | None)
        Names for each of the species.
    number_operators : 3D array
        The vector of occupation number operators with shape ``(dim, dim, num_species)``.
    n_tot : 2D array
        The total occupation number operator with shape ``(dim, dim)``.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 creators: np.ndarray,
                 annihilators: np.ndarray,
                 anti_commute_sign: Literal[+1, -1],
                 species_names: Sequence[str | None] = None,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        self.num_species = num_species = creators.shape[2]
        assert creators.shape == annihilators.shape == (leg.dim, leg.dim, num_species)
        self.creators = as_immutable_array(creators)
        self.annihilators = as_immutable_array(annihilators)
        self.anti_commute_sign = anti_commute_sign
        if species_names is None:
            species_names = [None] * num_species
        else:
            assert len(species_names) == num_species
        self.species_names = species_names
        self._species_name_to_idx = {name: idx for idx, name in enumerate(species_names)}

        # [p, (p*), k] @ [(p), p*, k] -> [p, p*, k]
        self.number_operators = n_ops = as_immutable_array(np.tensordot(creators, annihilators, (1, 0)))
        self.n_tot = as_immutable_array(np.sum(n_ops, axis=2))
        super().__init__(leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
                         backend=backend, default_device=default_device)

    def test_sanity(self):
        super().test_sanity()
        for i in range(self.num_species):
            N_i = self.number_operators[:, :, i]
            assert np.allclose(self.creators[:, :, i] @ self.annihilators[:, :, i], N_i)

            assert np.allclose(np.diag(np.diag(N_i)), N_i), 'expected diagonal N_i'
            assert np.allclose(np.around(N_i, 0), N_i), 'expected integer entries for N_i'
            assert np.all(N_i >= 0), 'expected non-negative entries for N_i'

            # check (anti-)commutation with same species
            BBd = self.annihilators[:, :, i] @ self.creators[:, :, i]
            assert np.allclose(BBd + self.anti_commute_sign * N_i, np.eye(self.leg.dim))

            # check commutation relations among different species
            # note: even for fermions, these numpy representations without explicit JW commute
            #       instead of anti-commuting.
            for j in range(i):
                BiBdj = self.annihilators[:, :, i] @ self.creators[:, :, j]
                BdjBi = self.creators[:, :, j] @ self.annihilators[:, :, i]
                assert np.allclose(BiBdj, BdjBi)
                BiBj = self.annihilators[:, :, i] @ self.annihilators[:, :, j]
                BjBi = self.annihilators[:, :, j] @ self.annihilators[:, :, i]
                assert np.allclose(BiBj, BjBi)
                BdiBdj = self.creators[:, :, i] @ self.creators[:, :, j]
                BdjBdi = self.creators[:, :, j] @ self.creators[:, :, i]
                assert np.allclose(BdiBdj, BdjBdi)

    def add_individual_occupation_ops(self):
        """Add occupation and parity operators for each species as symmetric onsite operators.

        The added operators include::
            - occupation operators ``Ni`` for each species ``i``
            - parity operators ``Pi`` for each species ``i``

        If there is only a single species, also the aliases ``N`` for ``N0`` and ``P`` for ``P0``.
        """
        for i in range(self.num_species):
            N_i = self.number_operators[:, :, i]
            self.add_onsite_operator(f'N{i}', N_i, is_diagonal=True)
        if self.num_species == 1:
            self.add_onsite_operator('N', self.onsite_operators['N0'])

    def add_total_occupation_ops(self):
        """Add total occupation and parity operators as symmetric onsite operators.

        The added operators include:
        - total occupation operator `Ntot`
        - total parity operator `Ptot`
        - squared total occupation operator `NtotNtot`
        """
        self.add_onsite_operator('Ntot', self.n_tot, is_diagonal=True)
        self.add_onsite_operator('NtotNtot', self.n_tot @ self.n_tot, is_diagonal=True)
        P_tot = np.diag(1. - 2. * np.mod(np.diag(self.n_tot), 2))
        self.add_onsite_operator('Ptot', P_tot, is_diagonal=True)

    @abstractmethod
    def get_annihilator_numpy(self, species: int | str, include_JW: bool = False):
        """Wrapper around ``annihilators[:, :, species]``, optionally including JW strings.

        If `include_JW`, we include the ``(-1) ** n_k`` from all ``k < species``.
        """
        ...

    @abstractmethod
    def get_creator_numpy(self, species: int | str, include_JW: bool = False):
        """Wrapper around ``creators[:, :, species]``, optionally including JW strings.

        If `include_JW`, we include the ``(-1) ** n_k`` from all ``k < species``.
        """
        ...

    def get_occupation_numpy(self, species: int | str | Sequence[int | str] = ALL_SPECIES):
        """Get the occupation number operator for some or multiple species as a numpy array."""
        if species is ALL_SPECIES:
            species = [*range(self.num_species)]
        else:
            species = [self.get_species_idx(s) for s in to_iterable(species)]
        return np.sum(self.number_operators[:, :, species], axis=2)

    def get_species_idx(self, species: int | str | None) -> int:
        if isinstance(species, str):
            species = self._species_name_to_idx[species]
        if species is None:
            if self.num_species > 1:
                raise ValueError('Need to specify the species')
            species = 0
        return to_valid_idx(species, self.num_species)


class BosonicDOF(OccupationDOF):
    """Common base class for sites that have a bosonic degree of freedom.

    Requires that the local basis is such that the :attr:`number_operators` of all species
    are diagonal.

    Mutually exclusive with :class:`FermionicDOF`. Sites containing both bosonic and fermionic
    degrees of freedom can be realized by grouping of a bosonic site with a fermionic one.

    Attributes
    ----------
    Nmax : 1D array of int
        Cutoff defining the maximum number of bosons per species and site. ``Nmax[i]`` corresponds
        to the cutoff for the `i`th species; a value of ``Nmax[i] = 1`` describes hard-core bosons.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 creators: np.ndarray,
                 annihilators: np.ndarray,
                 species_names: Sequence[str | None] = None,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        if isinstance(self, FermionicDOF):
            raise SymmetryError('FermionicDOF and BosonicDOF are incompatible.')
        OccupationDOF.__init__(
            self, leg, creators=creators, annihilators=annihilators, anti_commute_sign=+1,
            species_names=species_names, state_labels=state_labels,
            onsite_operators=onsite_operators, backend=backend, default_device=default_device
        )

        Nmax = []
        for i in range(self.num_species):
            N_i = self.n_ops[:, :, i]
            N_i_max_ = np.max(np.diag(N_i))
            N_i_max = round(N_i_max_, 0)
            assert np.allclose(N_i_max, N_i_max_)
            assert leg.dim % (N_i_max + 1) == 0
            Nmax.append(N_i_max)
        Nmax = np.asarray(Nmax, dtype=int)
        assert np.min(Nmax) > 0, (f'Invalid Nmax: {Nmax}; each boson species must have a max. '
                                  'occupation number of at least 1')
        self.Nmax = Nmax

    def test_sanity(self):
        super().test_sanity()
        for i in range(self.num_species):
            N_i = self.number_operators[:, :, i]
            # check commutation relations
            # BBd is 0 when going over the maximum occupation -> set this manually here
            BBd = self.annihilators[:, :, i] @ self.creators[:, :, i]
            mask = np.isclose(np.diag(BBd), 0)
            BBd[mask, mask] += self.Nmax[i] + 1
            assert np.allclose(BBd - N_i, np.eye(self.leg.dim))
            # N_i has integer eigenvalues and is diagonal
            N_i_rounded = np.around(N_i, 0)
            assert np.allclose(N_i_rounded, N_i)
            assert np.allclose(np.diag(np.diag(N_i)), N_i)
            assert np.min(N_i_rounded) == 0
            assert np.max(N_i_rounded) == self.Nmax[i]

    def add_individual_occupation_ops(self):
        OccupationDOF.add_individual_occupation_ops(self)
        for i in range(self.num_species):
            N_i = self.number_operators[:, :, i]
            P_i = np.diag(1. - 2. * np.mod(np.diag(N_i), 2))
            self.add_onsite_operator(f'N{i}N{i}', N_i @ N_i, is_diagonal=True)
            self.add_onsite_operator(f'P{i}', P_i, is_diagonal=True)
        if self.num_species == 1:
            self.add_onsite_operator('NN', self.onsite_operators['N0N0'])
            self.add_onsite_operator('P', self.onsite_operators['P0'])

    def get_annihilator_numpy(self, species, include_JW=False):
        return self.annihilators[:, :, self.get_species_idx(species)]

    def get_creator_numpy(self, species, include_JW=False):
        return self.creators[:, :, self.get_species_idx(species)]

    @staticmethod
    def conservation_law_to_symmetry(conserve: Literal['N', 'parity', 'None'] | Sequence[Literal['N', 'parity', 'None']]
                                     ) -> Symmetry | ProductSymmetry:
        """Translate conservation law for individual / all bosons to a symmetry."""
        if isinstance(conserve, str) or conserve is None:
            if conserve in ['N', 'Ntot', 'N_tot', 'U(1)', 'U1']:
                sym = U1Symmetry('total_occupation')
            elif conserve in ['parity', 'P', 'Ptot', 'P_tot', 'Z_2', 'Z2']:
                sym = ZNSymmetry(2, 'total_occupation_parity')
            elif conserve in ['None', 'none', None]:
                sym = NoSymmetry()
            else:
                raise ValueError(f'Invalid `conserve`: {conserve}')
        elif is_iterable(conserve):
            sym_factors = []
            num_no_sym = 0
            for i, conserve_i in enumerate(conserve):
                if conserve_i in ['N', 'Ni', 'N_i', 'U(1)', 'U1']:
                    sym_factors.append(U1Symmetry(f'species{i}_occupation'))
                elif conserve_i in ['parity', 'P', 'Pi', 'P_i', 'Z_2', 'Z2']:
                    sym_factors.append(ZNSymmetry(2, f'species{i}_occupation_parity'))
                elif conserve_i in ['None', 'none', None]:
                    sym_factors.append(NoSymmetry())
                    num_no_sym += 1
                else:
                    raise ValueError(f'Invalid entry in `conserve`: {conserve_i}')
            if num_no_sym == len(conserve):
                sym = NoSymmetry()
            else:
                sym = ProductSymmetry(sym_factors)
        else:
            raise ValueError(f'Invalid `conserve`: {conserve}')
        return sym

    @staticmethod
    def _states_with_occupation(n: int, Nmax: list[int] | np.ndarray) -> int:
        """Number of states with a given total boson number for given maximum occupations."""
        if len(Nmax) == 1:
            if n <= Nmax[0]:
                return 1
            return 0
        # lower and upper bounds on the first species occupation such that n can still be reached
        lower_bound = max([0, n - sum(Nmax[1:])])
        upper_bound = max([0, n - Nmax[0]])
        num_states = np.sum([BosonicDOF._states_with_occupation(n_1, Nmax[1:])
                             for n_1 in range(upper_bound, n + 1 - lower_bound)])
        return num_states

    @staticmethod
    def _creation_annihilation_op_from_single_Nmax(Nmax: int) -> tuple[np.ndarray, np.ndarray]:
        """Construct the creation and annihilation operators for a single boson."""
        assert isinstance(Nmax, (int, np.integer))
        assert Nmax > 0, f'Invalid Nmax: {Nmax}; bosons must have a max. occupation number of at least 1'
        dim = Nmax + 1
        B = np.zeros([dim, dim], dtype=np.float64)
        for n in range(1, dim):
            B[n - 1, n] = np.sqrt(n)
        return np.transpose(B), B

    @staticmethod
    def _creation_annihilation_ops_from_Nmax(Nmax: list[int] | np.ndarray
                                             ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the creation and annihilation operators for multiple boson species."""
        Nmax_ = np.asarray(Nmax, dtype=int)
        assert np.allclose(Nmax_, Nmax), f'Invalid `Nmax`: {Nmax}'
        creators_i = []
        annihilators_i = []
        for N in Nmax_:
            Bd_i, B_i = BosonicDOF._creation_annihilation_op_from_single_Nmax(N)
            creators_i.append(Bd_i)
            annihilators_i.append(B_i)
        ids_i = [np.eye(N + 1) for N in Nmax_]
        creators = []
        annihilators = []
        for i in range(len(Nmax_)):
            creators.append(reduce(np.kron, [*ids_i[:i], creators_i[i], *ids_i[i + 1:]]))
            annihilators.append(reduce(np.kron, [*ids_i[:i], annihilators_i[i], *ids_i[i + 1:]]))
        creators = np.stack(creators, axis=2)
        annihilators = np.stack(annihilators, axis=2)
        return creators, annihilators


class FermionicDOF(OccupationDOF):
    """Common base class for sites that have a fermionic degree of freedom.

    Requires that the local basis is such that the :attr:`number_operators` of all species
    are diagonal.

    Mutually exclusive with :class:`BosonicDOF`. Sites containing both bosonic and fermionic
    degrees of freedom can be realized by grouping of a bosonic site with a fermionic one.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 creators: np.ndarray,
                 annihilators: np.ndarray,
                 species_names: Sequence[str | None] = None,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        if isinstance(leg.symmetry, ProductSymmetry):
            # there should only be a single fermionic symmetry
            assert sum([isinstance(factor, (FermionParity, FermionNumber)) for factor in leg.symmetry.factors]) == 1
        else:
            assert isinstance(leg.symmetry, (FermionParity, FermionNumber))
        if isinstance(self, BosonicDOF):
            raise SymmetryError('FermionicDOF and BosonicDOF are incompatible.')
        OccupationDOF.__init__(
            self, leg=leg, creators=creators, annihilators=annihilators, anti_commute_sign=-1,
            species_names=species_names, state_labels=state_labels,
            onsite_operators=onsite_operators, backend=backend, default_device=default_device
        )

        n_diag = self.number_operators[np.arange(self.dim), np.arange(self.dim), :]  # [p, k]
        n_before = np.cumsum(n_diag, axis=1)  # \sum_{q < k} n_k
        partial_JW = np.zeros((self.dim, self.dim, self.num_species))
        partial_JW[np.arange(self.dim), np.arange(self.dim), :] = (-1) ** n_before
        self._partial_JWs = as_immutable_array(partial_JW)
        self._JW = as_immutable_array(np.diag((-1) ** np.diag(self.n_tot)))

        for k in range(self.num_species):
            N_k_max_ = np.max(np.diag(self.number_operators[:, :, k]))
            N_k_max = round(N_k_max_, 0)
            assert np.allclose(N_k_max, N_k_max_)
            assert N_k_max == 1

    def test_sanity(self):
        super().test_sanity()
        for i in range(self.num_species):
            N_i = self.number_operators[:, :, i]
            # check fermions square to zero
            CC = self.annihilators[:, :, i] @ self.annihilators[:, :, i]
            assert np.allclose(CC, np.zeros_like(CC))
            CdCd = self.creators[:, :, i] @ self.creators[:, :, i]
            assert np.allclose(CdCd, np.zeros_like(CdCd))
            # check Pauli exclusion
            assert np.max(N_i) <= 1, 'expect entries <= 1 for N_i'

    def get_annihilator_numpy(self, species: int, include_JW: bool = False):
        species = self.get_species_idx(species)
        res = self.annihilators[:, :, species]
        if include_JW:
            res = res @ self._partial_JWs[:, :, species]
        return res

    def get_creator_numpy(self, species: int, include_JW: bool = False):
        species = self.get_species_idx(species)
        res = self.creators[:, :, species]
        if include_JW:
            res = res @ self._partial_JWs[:, :, species]
        return res

    @staticmethod
    def conservation_law_to_symmetry(conserve: Literal['N', 'parity'] | Sequence[Literal['N', 'parity', 'None']]
                                     ) -> Symmetry | ProductSymmetry:
        """Translate conservation law for individual / all fermions to a symmetry."""
        if isinstance(conserve, str):
            if conserve in ['N', 'Ntot', 'N_tot']:
                sym = ProductSymmetry([U1Symmetry('total_fermion_occupation'),
                                       FermionParity('total_fermion_parity')])
            elif conserve in ['parity', 'P', 'Ptot', 'P_tot']:
                sym = FermionParity('total_fermion_parity')
            else:
                raise ValueError(f'Invalid `conserve`: {conserve}')
        elif is_iterable(conserve):
            sym_factors = []
            num_no_sym = 0
            for i, conserve_i in enumerate(conserve):
                if conserve_i in ['N', 'Ni', 'N_i']:
                    sym_factors.append(U1Symmetry(f'species{i}_fermion_occupation'))
                elif conserve_i in ['parity', 'P', 'Pi', 'P_i']:
                    sym_factors.append(ZNSymmetry(2, f'species{i}_fermion_parity'))
                elif conserve_i in ['None', 'none', None]:
                    sym_factors.append(NoSymmetry())
                    num_no_sym += 1
                else:
                    raise ValueError(f'Invalid entry in `conserve`: {conserve_i}')
            if num_no_sym == len(conserve):
                sym = FermionParity('total_fermion_parity')
            else:
                sym = ProductSymmetry([*sym_factors, FermionParity('total_fermion_parity')])
        else:
            raise ValueError(f'Invalid `conserve`: {conserve}')
        return sym

    @staticmethod
    def _states_with_occupation(n: int, num_species: int) -> int:
        """Number of states with a given total fermion number for given number of species."""
        return comb(num_species, n)

    @staticmethod
    def _creation_annihilation_ops(num_species: int) -> tuple[np.ndarray, np.ndarray]:
        """Construct the creation and annihilation operators for multiple fermion species."""
        return BosonicDOF._creation_annihilation_ops_from_Nmax([1] * num_species)


class ClockDOF(Site):
    """Common base class for sites that have a quantum clock degree of freedom.

    Attributes
    ----------
    q : int
        Number of clock states. A quantum clock reduces to a spin-1/2 if ``q == 2``.
    clock_operators : 3D array
        The vector of clock operators ``X`` and ``Z`` as a numpy array with axes ``[p, p*, i]``
        and shape ``(dim, dim, 2)``.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 q: int,
                 clock_operators: np.ndarray,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        self.q = q
        assert clock_operators.shape == (leg.dim, leg.dim, 2)
        assert leg.dim % q == 0
        self.clock_operators = as_immutable_array(clock_operators)

        super().__init__(
            leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )

        Z = clock_operators[:, :, 1]
        Zhc = np.conj(clock_operators[:, :, 1].T)
        self.add_onsite_operator('Z', Z, is_diagonal=True)
        self.add_onsite_operator('Zhc', Zhc, is_diagonal=True)
        self.add_onsite_operator('Zphc', Z + Zhc, is_diagonal=True)

    def test_sanity(self):
        super().test_sanity()
        # check commutation relations
        X, Z = [self.clock_operators[:, :, i] for i in range(2)]
        Xhc, Zhc = [np.conj(self.clock_operators[:, :, i].T) for i in range(2)]
        assert np.allclose(X @ Z, np.exp(2.j * np.pi / self.q) * Z @ X)

        identity = np.eye(X.shape[0])
        assert np.allclose(np.linalg.matrix_power(X, self.q), identity)
        assert np.allclose(np.linalg.matrix_power(Z, self.q), identity)
        assert np.allclose(X @ Xhc, identity)
        assert np.allclose(Z @ Zhc, identity)


class AnyonDOF(Site):
    """Common base class for sites that have an anyonic degree of freedom.

    Parameters
    ----------
    sector_names : sequence of str or None
        The sector names that appear in the onsite projection operators. The `i`th operator is
        called `f'P_{sector_names[i]}'` and projects onto the `i`th sector in
        `leg.sector_decomposition`. For `None` entries (default), no projection operators are
        constructed.
    """

    def __init__(self, leg: ElementarySpace, state_labels: dict[str, int] = None,
                 sector_names: Sequence[str | None] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None, default_device: str = None):
        if sector_names is None:
            sector_names = [None] * leg.num_sectors
        assert len(sector_names) == leg.num_sectors
        if onsite_operators is None:
            onsite_operators = {}
        self.sector_names = sector_names
        for sector, sector_name in zip(leg.sector_decomposition, sector_names):
            if sector_name is None:
                continue
            P_sec = sector_proj_onsite(leg.symmetry, leg, sector,
                                       backend=backend, device=default_device)
            onsite_operators[f'P_{sector_name}'] = P_sec
        super().__init__(
            leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )


def consistent_leg_symmetry(leg: ElementarySpace, symmetry_factor: Symmetry,
                            symmetry_sector_slice: slice) -> bool:
    """Test whether the symmetry of a leg contains a certain factor at a given slice.

    Parameters
    ----------
    leg : ElementarySpace
        The leg whose symmetry is tested.
    symmetry_factor : Symmetry
        The symmetry that should be contained in `leg.symmetry`. If `leg.symmetry` is not
        a `ProductSymmetry`, it is tested whether ``leg.symmetry == symmetry_factor``.
    symmetry_sector_slice : slice
        A slice such that the entries ``leg.sector_decomposition[:, slc]`` correspond to
        `symmetry_factor`.
    """
    slc = symmetry_sector_slice
    slc_start = 0 if slc.start is None else slc.start
    if isinstance(symmetry_factor, ProductSymmetry):
        if not isinstance(leg.symmetry, ProductSymmetry):
            return False
        slc_stop = leg.symmetry.sector_slices[-1] if slc.stop is None else slc.stop
        if len(leg.symmetry.factors) == len(symmetry_factor.factors):
            # factors equal and slice over full symmetry
            return (leg.symmetry.is_same_symmetry(symmetry_factor) and
                    slc_start == 0 and slc_stop == leg.symmetry.sector_slices[-1])
        factor_idx = np.searchsorted(leg.symmetry.sector_slices, slc_start)
        slc_bounds = slc_start + symmetry_factor.sector_slices
        leg_slc_bounds = leg.symmetry.sector_slices[factor_idx:factor_idx + len(symmetry_factor.factors)]
        return (all(leg_slc_bounds == slc_bounds) and
                all([leg.symmetry.factors[factor_idx + i] == factor
                     for i, factor in enumerate(symmetry_factor.factors)]))

    # remaining part: symmetry_factor is not a ProductSymmetry
    if isinstance(leg.symmetry, ProductSymmetry):
        # symmetry_sector_slice tells us which factor in the leg symmetry is symmetry_factor
        slc_stop = leg.symmetry.sector_slices[-1] if slc.stop is None else slc.stop
        factor_idx = np.searchsorted(leg.symmetry.sector_slices, slc_start)
        return all([slc_start == leg.symmetry.sector_slices[factor_idx],
                    slc_stop == leg.symmetry.sector_slices[factor_idx + 1],
                    leg.symmetry.factors[factor_idx] == symmetry_factor])
    # leg symmetry is not a ProductSymmetry; slc must still match sector_ind_len
    slc_stop = leg.symmetry.sector_ind_len if slc.stop is None else slc.stop
    return leg.symmetry == symmetry_factor and slc_start == 0 and slc_stop == leg.symmetry.sector_ind_len


def sector_proj_onsite(symmetry: Symmetry, leg: ElementarySpace, sector: Sector,
                       backend: TensorBackend = None, device: str = None):
    """Helper function to create onsite projectors onto sectors"""
    assert symmetry.is_valid_sector(sector)
    return SymmetricTensor.from_sector_projection([leg], sector, labels=['p', 'p*'],
                                                  backend=backend, device=device)
