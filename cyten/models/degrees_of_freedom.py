"""Defines classes describing the local physical Hilbert spaces.

The :class:`DegreeOfFreedom` is the prototype, read its docstring.
All other classes are base classes from which sites are derived.
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np
from typing import Sequence
from functools import reduce
from math import comb

from ..backends import TensorBackend, get_backend
from ..backends.abstract_backend import Block
from ..spaces import ElementarySpace
from ..tensors import SymmetricTensor
from ..symmetries import (
    FermionNumber, FermionParity, Sector, Symmetry, ProductSymmetry, BraidingStyle
)


class DegreeOfFreedom:
    """Collects necessary information about a local degree of freedom of a lattice model.

    A degree of freedom defines the local Hilbert space in terms of its :attr:`leg`.
    This involves a choice for the local basis.
    Moreover, it exposes the symmetric single-site operators.
    Multi-site operators, on the other hand, are represented by :class:`Coupling` s.

    .. todo ::
        How to handle JW strings?

    .. todo ::
        Mechanism to change the symmetry

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

    def add_onsite_operator(self, name: str, op: SymmetricTensor | Block | Sequence[Sequence[float]],
                            understood_braiding: bool = False):
        """Add an operator to the :attr:`onsite_operators`."""
        if name in self.onsite_operators:
            return  # TODO warn? error?
        #
        if isinstance(op, SymmetricTensor):
            assert op.codomain.factors == [self.leg]
            assert op.domain.factors == [self.leg]
            if op.labels != ['p', 'p*']:
                op = op.copy(deep=False)
                op.labels = ['p', 'p*']
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


class SpinDOF(DegreeOfFreedom):
    """Common base class for sites that have a spin degree of freedom.

    Attributes
    ----------
    spin_vector : 3D array
        The vector of spin operators as a numpy array with axes ``[p, p*, i]`` and shape
        ``(dim, dim, 3)``. These operators include the factor of the total spin, i.e., the largest
        eigenvalue of any of the ``spin_vector[:, :, i]`` is :attr:`total_spin`.
        E.g., for spin-1/2, these are ``.5`` times the pauli matrices.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 spin_vector: np.ndarray,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        assert spin_vector.shape == (leg.dim, leg.dim, 3)
        self.spin_vector = spin_vector
        DegreeOfFreedom.__init__(
            self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
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
    def _spin_vector_from_Sp(Sz: np.ndarray, Sp: np.ndarray) -> np.ndarray:
        """Build the spin_vector from ``Sz`` and ``Sp = Sx + i Sy``"""
        dim = Sz.shape[0]
        assert Sz.shape == (dim, dim)
        assert Sp.shape == (dim, dim)
        Sm = Sp.T.conj()
        Sx = .5 * (Sp + Sm)
        Sy = .5j * (Sm - Sp)
        return np.stack([Sx, Sy, Sz], axis=-1)


class BosonicDOF(DegreeOfFreedom):
    """Common base class for sites that have a bosonic degree of freedom.

    TODO find a good format to doc the onsite operators that exist in a site

    Attributes
    ----------
    num_species : int
        Number of boson species.
    Nmax : 1D array of int
        Cutoff defining the maximum number of bosons per species and site. ``Nmax[i]`` corresponds
        to the cutoff for the `i`th species; a value of ``Nmax[i] = 1`` describes hard-core bosons.
    creators : 3D array
        The vector of creation operators as a numpy array with shape ``(dim, dim, num_species)``
        and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
        ``[Bd0, Bd1`, ...]`` stacked along axis 2).
    annihilators : 3D array
        The vector of annihilation operators as a numpy array with shape ``(dim, dim, num_species)``
        and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
        ``[B0, B1`, ...]`` stacked along axis 2).
    """

    def __init__(self,
                 leg: ElementarySpace,
                 creators: np.ndarray,
                 annihilators: np.ndarray,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        assert creators.shape[:2] == (leg.dim, leg.dim)
        assert creators.shape == annihilators.shape
        self.creators = creators
        self.annihilators = annihilators
        self.num_species = num_species = creators.shape[2]

        Nmax = []
        N_is = []  # these are used later for the on-site operators
        for i in range(num_species):
            N_i = creators[:, :, i] @ annihilators[:, :, i]
            N_i_max_ = np.max(np.diag(N_i))
            N_i_max = round(N_i_max_, 0)
            assert np.allclose(N_i_max, N_i_max_)
            assert leg.dim % (N_i_max + 1) == 0
            N_is.append(N_i)
            Nmax.append(N_i_max)
        Nmax = np.asarray(Nmax, dtype=int)
        assert np.min(Nmax) > 0, (f'Invalid Nmax: {Nmax}; each boson species must have a max. '
                                  'occupation number of at least 1')
        self.Nmax = Nmax

        DegreeOfFreedom.__init__(
            self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )

        # TODO this should work for any symmetry
        if num_species > 1:
            # operator names include a number for the species or 'tot' for total
            N_iN_is = []
            P_is = []
            for i in range(num_species):
                self.add_onsite_operator(f'N{i}', N_is[i])
                N_iN_i = np.diag(np.diag(N_is[i]) ** 2)
                self.add_onsite_operator(f'N{i}N{i}', N_iN_i)
                N_iN_is.append(N_iN_i)
                P_i = np.diag(1. - 2. * np.mod(np.diag(N_is[i]), 2))
                self.add_onsite_operator(f'P{i}', P_i)
                P_is.append(P_i)
            N_tot = np.sum(N_is, axis=0)
            self.add_onsite_operator('Ntot', N_tot)
            self.add_onsite_operator('NtotNtot', np.diag(np.diag(N_tot) ** 2))
            P_tot = np.diag(1. - 2. * np.mod(np.diag(N_tot), 2))
            self.add_onsite_operator('Ptot', P_tot)
        else:
            self.add_onsite_operator('N', N_is[0])
            NN = np.diag(np.diag(N_is[0]) ** 2)
            self.add_onsite_operator('NN', NN)
            P = np.diag(1. - 2. * np.mod(np.diag(N_is[0]), 2))
            self.add_onsite_operator('P', P)

    def test_sanity(self):
        super().test_sanity()
        for i in range(self.num_species):
            N_i = self.creators[:, :, i] @ self.annihilators[:, :, i]
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

            # check commutation relations among different species
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

    @staticmethod
    def _states_with_occupation(n: int, Nmax: list[int] | np.ndarray) -> int:
        """Number of states with a given total boson number for given maximum occupations."""
        if len(Nmax) == 1:
            if n <= Nmax[0]:
                return 1
            return 0
        # lower and upper bounds on the first species occuption such that n can still be reached
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
    def _creation_annihilation_ops_from_Nmax(Nmax: list[int] | np.ndarray[int]
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


class FermionicDOF(DegreeOfFreedom):
    """Common base class for sites that have a fermionic degree of freedom.

    TODO onsite operators

    Attributes
    ----------
    num_species : int
        Number of fermion species.
    creators : 3D array
        The vector of creation operators as a numpy array with shape ``(dim, dim, num_species)``
        and axes ``[p, p*, i]``, where `i` corresponds to the different species of fermions (i.e.,
        ``[Cd0, Cd1`, ...]`` stacked along axis 2).
    annihilators : 3D array
        The vector of annihilation operators as a numpy array with shape ``(dim, dim, num_species)``
        and axes ``[p, p*, i]``, where `i` corresponds to the different species of fermions (i.e.,
        ``[C0, C1`, ...]`` stacked along axis 2).
    """

    creators: np.ndarray  # [p, p*, i] where i are different species ;  == [Cd0, Cd1, ...]
    annihilators: np.ndarray  # [p, p*, i] ;  == [C0, C1, ...]

    def __init__(self,
                 leg: ElementarySpace,
                 creators: np.ndarray,
                 annihilators: np.ndarray,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None,
                 backend: TensorBackend = None,
                 default_device: str = None):
        if isinstance(leg.symmetry, ProductSymmetry):
            # there should only be a single fermionic symmetry
            assert sum([isinstance(factor, (FermionParity, FermionNumber)) for factor in leg.symmetry.factors]) == 1
        else:
            assert isinstance(leg.symmetry, (FermionParity, FermionNumber))
        assert creators.shape[:2] == (leg.dim, leg.dim)
        assert creators.shape == annihilators.shape
        self.creators = creators
        self.annihilators = annihilators
        self.num_species = num_species = creators.shape[2]
        assert leg.dim % (2 ** num_species) == 0

        N_is = []  # these are used later for the on-site operators
        for i in range(num_species):
            N_i = creators[:, :, i] @ annihilators[:, :, i]
            N_i_max_ = np.max(np.diag(N_i))
            N_i_max = round(N_i_max_, 0)
            assert np.allclose(N_i_max, N_i_max_)
            assert N_i_max == 1
            N_is.append(N_i)

        DegreeOfFreedom.__init__(
            self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )

        # TODO this should work for any symmetry
        if num_species > 1:
            # operator names include a number for the species or 'tot' for total
            for i in range(num_species):
                self.add_onsite_operator(f'N{i}', N_is[i], understood_braiding=True)
            N_tot = np.sum(N_is, axis=0)
            self.add_onsite_operator('Ntot', N_tot, understood_braiding=True)
            self.add_onsite_operator('NtotNtot', np.diag(np.diag(N_tot) ** 2), understood_braiding=True)
            P_tot = np.diag(np.mod(np.diag(N_tot), 2))
            self.add_onsite_operator('Ptot', P_tot, understood_braiding=True)
        else:
            # no parity here since parity == occupation for a single species
            self.add_onsite_operator('N', N_is[0], understood_braiding=True)

    def test_sanity(self):
        super().test_sanity()
        for i in range(self.num_species):
            N_i = self.creators[:, :, i] @ self.annihilators[:, :, i]
            # check anticommutation relations
            CCd = self.annihilators[:, :, i] @ self.creators[:, :, i]
            assert np.allclose(CCd + N_i, np.eye(self.leg.dim))
            CC = self.annihilators[:, :, i] @ self.annihilators[:, :, i]
            assert np.allclose(CC, np.zeros_like(CC))
            CdCd = self.creators[:, :, i] @ self.creators[:, :, i]
            assert np.allclose(CdCd, np.zeros_like(CdCd))
            # N_i has integer eigenvalues and is diagonal
            N_i_rounded = np.around(N_i, 0)
            assert np.allclose(N_i_rounded, N_i)
            assert np.allclose(np.diag(np.diag(N_i)), N_i)
            assert np.min(N_i_rounded) == 0
            assert np.max(N_i_rounded) == 1

            # check anticommutation relations among different species
            for j in range(i):
                # on this level, JW strings are necessary to make the operators anticommuting
                # -> check commutation instead
                CiCdj = self.annihilators[:, :, i] @ self.creators[:, :, j]
                CdjCi = self.creators[:, :, j] @ self.annihilators[:, :, i]
                assert np.allclose(CiCdj, CdjCi)
                CiCj = self.annihilators[:, :, i] @ self.annihilators[:, :, j]
                CjCi = self.annihilators[:, :, j] @ self.annihilators[:, :, i]
                assert np.allclose(CiCj, CjCi)
                CdiCdj = self.creators[:, :, i] @ self.creators[:, :, j]
                CdjCdi = self.creators[:, :, j] @ self.creators[:, :, i]
                assert np.allclose(CdiCdj, CdjCdi)

    @staticmethod
    def _states_with_occupation(n: int, num_species: int) -> int:
        """Number of states with a given total fermion number for given number of species."""
        return comb(num_species, n)

    @staticmethod
    def _creation_annihilation_ops(num_species: int) -> tuple[np.ndarray, np.ndarray]:
        """Construct the creation and annihilation operators for multiple fermion species."""
        return BosonicDOF._creation_annihilation_ops_from_Nmax([1] * num_species)


class ClockDOF(DegreeOfFreedom):
    """Common base class for sites that have a quantum clock degree of freedom.

    TODO onsite operators

    Attributes
    ----------
    q : int
        Number of states per site.
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
        self.clock_operators = clock_operators

        DegreeOfFreedom.__init__(
            self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )

        # TODO this should work for any symmetry
        Z = clock_operators[:, :, 1]
        Zhc = np.conj(clock_operators[:, :, 1].T)
        self.add_onsite_operator('Z', Z)
        self.add_onsite_operator('Zhc', Zhc)
        self.add_onsite_operator('Zphc', Z + Zhc)

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


class RepresentationDOF(DegreeOfFreedom):
    """Common base class for sites that have a degree of freedom described by a category.

    TODO onsite operators

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
        DegreeOfFreedom.__init__(
            self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators,
            backend=backend, default_device=default_device
        )


def consistent_leg_symmetry(leg: ElementarySpace, symmetry_factor: Symmetry,
                            symmetry_sector_slice: slice) -> bool:
    """Test whether the symmetry of a leg contains a certain factor at a given slice.

    TODO do we still need this?

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
    # TODO is this a reason to say that we should always work with ProductSymmetries?
    #      like tenpy v1, we always have a list of qmod, even if there is only 1
    # leg symmetry is not a ProductSymmetry; slc must still match sector_ind_len
    slc_stop = leg.symmetry.sector_ind_len if slc.stop is None else slc.stop
    return leg.symmetry == symmetry_factor and slc_start == 0 and slc_stop == leg.symmetry.sector_ind_len


def get_same_DOF_backend(*dofs: DegreeOfFreedom, error_msg: str = 'Incompatible backends.') -> TensorBackend:
    """If the given tensors have the same backend, return it. Raise otherwise."""
    if len(dofs) == 0:
        raise ValueError('Need at least one tensor')
    backend = dofs[0].backend
    if not all(dof.backend == backend for dof in dofs[1:]):
        raise ValueError(error_msg)
    return backend


def get_same_DOF_device(*dofs: DegreeOfFreedom, error_msg: str = 'Incompatible devices.') -> str:
    """If the given tensors have the same device, return it. Raise otherwise."""
    if len(dofs) == 0:
        raise ValueError('Need at least one tensor')
    device = dofs[0].default_device
    if not all(dof.default_device == device for dof in dofs[1:]):
        raise ValueError(error_msg)
    return device


def sector_proj_onsite(symmetry: Symmetry, leg: ElementarySpace, sector: Sector,
                       backend: TensorBackend = None, device: str = None):
    """Helper function to create onsite projectors onto sectors"""
    assert symmetry.is_valid_sector(sector)
    return SymmetricTensor.from_sector_projection([leg], sector, labels=['p', 'p*'],
                                                  backend=backend, device=device)
