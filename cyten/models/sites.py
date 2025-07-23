"""Defines a class describing the local physical Hilbert space.

The :class:`Site` is the prototype, read its docstring.
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np
from typing import Literal, Sequence

from ..backends import TensorBackend, get_backend
from ..backends.abstract_backend import Block
from ..spaces import ElementarySpace
from ..tensors import SymmetricTensor
from ..symmetries import (
    Symmetry, SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory,
    ProductSymmetry, BraidingStyle, FermionNumber, FermionParity
)


class Site:
    """Collects necessary information about a single local site of a lattice model.

    A site defines the local Hilbert space in terms of its :attr:`leg`.
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
        self.onsite_operators: dict[str, SymmetricTensor] = {}
        if onsite_operators is not None:
            for name, op in onsite_operators.items():
                self.add_op(name, op)
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        self.backend = backend
        self.default_device = default_device

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

    def add_onsite_operator(self, name: str, op: SymmetricTensor | Block | Sequence[Sequence[float]]):
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
                labels=['p', 'p*'], device=self.default_device
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


class SpinfulSite(Site):
    """Common base class for sites that have a spin degree of freedom.

    TODO find a good format to doc the onsite operators that exist in a site

    Attributes
    ----------
    double_total_spin : int
        Twice the :attr:`total_spin`. We store this, because it is an integer.
    spin_vector : 3D array
        The vector of spin operators as a numpy array with axes ``[p, p*, i]`` and shape
        ``(dim, dim, 3)``. These operators include the factor of the total spin, i.e., the largest
        eigenvalue of any of the ``spin_vector[:, :, i]`` is :attr:`total_spin`.
        E.g., for spin-1/2, these are ``.5`` times the pauli matrices.
    spin_symmetry : SU2Symmetry | U1Symmetry | ZNSymmetry | NoSymmetry
        The symmetry of the spin degree of freedom that is enforced.
        We can conserve::

            - SU(2), the full spin rotation symmetry
            - U(1), with sector labels corresponding to ``2 * Sz``
            - Z_2, with sector labels corresponding to ``(Sz + S_tot) % 2``.
            - nothing

        The full :attr:`symmetry` must either coincide with the `spin_symmetry`, or it must be
        a :class:`ProductSymmetry` with the `spin_symmetry` as a factor.
    spin_symmetry_sector_slice : slice
        A slice such that the entries ``leg.sector_decomposition[:, slc]`` correspond to the
        :attr:`spin_symmetry`.
    """

    def __init__(self,
                 leg: ElementarySpace,
                 double_total_spin: int,
                 spin_vector: np.ndarray,
                 spin_symmetry: SU2Symmetry | U1Symmetry | ZNSymmetry | NoSymmetry,
                 spin_symmetry_sector_slice: slice,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None):
        self.double_total_spin = double_total_spin
        assert spin_vector.shape == (leg.dim, leg.dim, 3)
        self.spin_vector = spin_vector

        # check the spin_symmetry, and that the sectors of the leg come in correct multiplets
        self.spin_symmetry_sector_slice = slc = spin_symmetry_sector_slice
        assert leg.dim % (double_total_spin + 1) == 0
        if isinstance(spin_symmetry, SU2Symmetry):
            # all must be in the same single spin sector
            assert np.all(leg.sector_decomposition[:, slc] == double_total_spin)
        elif isinstance(spin_symmetry, U1Symmetry):
            # make sure every Sz sector appears the same number of times.
            expect = leg.num_sectors // (double_total_spin + 1)
            for m in range(-double_total_spin, double_total_spin + 2, 2):
                num_sectors = np.sum(leg.sector_decomposition[:, slc] == m)
                assert num_sectors == expect
        elif isinstance(spin_symmetry, ZNSymmetry):
            assert spin_symmetry.N == 2
            num_m = double_total_spin + 1
            num_even_m = (num_m + 1) // 2
            num_odd_m = num_m // 2
            expect_even = (leg.num_sectors * num_even_m) // num_m
            expect_odd = (leg.num_sectors * num_odd_m) // num_m
            assert np.sum(leg.sector_decomposition[:, slc] == 0) == expect_even
            assert np.sum(leg.sector_decomposition[:, slc] == 1) == expect_odd
        elif isinstance(spin_symmetry, NoSymmetry):
            pass
        else:
            raise TypeError('Invalid spin_symmetry')
        # TODO do we want specific error messages when the slice is inconsistent?
        assert consistent_leg_symmetry(leg, spin_symmetry, slc)
        self.spin_symmetry = spin_symmetry

        Site.__init__(self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators)
        if not isinstance(spin_symmetry, SU2Symmetry):
            self.add_onsite_operator('Sz', spin_vector[:, :, 2])
        if isinstance(spin_symmetry, NoSymmetry):
            self.add_onsite_operator('Sx', spin_symmetry[:, :, 0])
            self.add_onsite_operator('Sy', spin_symmetry[:, :, 1])

    def test_sanity(self):
        super().test_sanity()
        # check commutation relations
        Sx, Sy, Sz = [self.spin_vector[:, :, i] for i in range(3)]
        assert np.allclose(Sx @ Sy - Sy @ Sx, 1j * Sz)
        assert np.allclose(Sy @ Sz - Sz @ Sy, 1j * Sx)
        assert np.allclose(Sz @ Sx - Sx @ Sz, 1j * Sy)
        # TODO check compatibility of the operators with the spin_symmetry, i.e. eigenvalues
        #      match with charge sectors
        S_sq = np.tensordot(self.spin_vector, self.spin_vector, ([-1, 1], [-1, 0]))
        eigenvalue = self.double_total_spin * (self.double_total_spin + 2) / 4
        assert np.allclose(S_sq, eigenvalue * np.eye(self.double_total_spin + 1))

    @property
    def total_spin(self) -> float:
        return self.double_total_spin / 2

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


class FermionicSite(Site):
    """TODO similar structure and role as SpinfulSite

    Mutually exclusive with BosonicSite, but compatible with SpinfulSite

    TODO how do we build

    Can conserve:
        - total fermion number sum_i N_i
        - fermion parity (sum_i N_i) % 2
        - TODO: how do we build a symmetry that conserves e.g. all N_i individually, but does the
                braiding w.r.t the parity of sum_i N_i.
                This is basically a fermionic special case of ProductSymmetry, no?
                (This means it might make sense to nest a fermionic product inside a ProductSymmetry
                 and currently ProductSymmetry assumes that there is no nesting...)
                The braiding depends on all fermionic particle numbers, so a single factor can
                not capture it alone!
        - TODO should we allow arbitrary combinations? e.g. we could conserve (N_1 + N_2, N_3),
               and we can mix conserving the particle number or just its parity, e.g. could do
               (N_1, N_2 % 2, N_3 + N_4, (N_5 + N_6) % 2) ...
               do we need that? how complicated/hard is it to do it this general?

    .. todo ::
        For now, assume that the symmetry needs to capture the fermionic statistics.
        Do not think about JW strings yet...
        That is also the reason why NoSymmetry is not an option here
    """

    creators: np.ndarray  # [p, p*, i] where i are different species ;  == [Cd0, Cd1, ...]
    annihilators: np.ndarray  # [p, p*, i] ;  == [C0, C1, ...]
    occupation_symmetry: FermionNumber | FermionParity  # TODO allow multiple


class BosonicSite(Site):
    """TODO similar to FermionicSite, but with bosons.

    Can conserve:
        - for each species either the particle number or its parity
        - TODO should we allow weirder combinations like ``N_1 + N_2`` and ``N_3`` ?
        - total particle number sum_i N_i
        - total parity (sum_i N_i) % 2

    """

    creators: np.ndarray  # [p, p*, i] where i are different species ;  == [Bd0, Bd1, ...]
    annihilators: np.ndarray  # [p, p*, i] ;  == [B0, B1, ...]
    occupation_symmetry: ProductSymmetry | U1Symmetry | ZNSymmetry | NoSymmetry


class SpinSite(SpinfulSite):
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

        SpinfulSite.__init__(self, leg=leg, double_total_spin=two_S, spin_vector=spin_vector,
                             spin_symmetry=sym, spin_symmetry_sector_slice=slice(None, None),
                             state_labels=state_labels)

    def __repr__(self):
        return f'SpinSite(S={self.S}, conserve={self.conserve})'


class SpinHalfFermionSite(SpinfulSite, FermionicSite):
    """TODO similar to SpinSite..."""


class ClockSite(Site):
    """Common base class for sites that have a quantum clock degree of freedom.

    TODO onsite operators

    Attributes
    ----------
    q : int
        Number of states per site.
    clock_operators : 3D array
        The vector of clock operators ``X`` and ``Z`` as a numpy array with axes ``[p, p*, i]``
        and shape ``(dim, dim, 2)``.
    clock_symmetry : ZNSymmetry | NoSymmetry
        The symmetry of the clock degree of freedom that is enforced.
        We can conserve::

            - Z_q, with sector label ``i`` corresponding to the states with eigenvalue
                ``exp(i * 2.j * pi / q)`` w.r.t. the diagonal on-site operator ``Z``.
            - nothing

        The full :attr:`symmetry` must either coincide with the `clock_symmetry`, or it must be
        a :class:`ProductSymmetry` with `clock_symmetry` as a factor.
    clock_symmetry_sector_slice : slice
        A slice such that the entries ``leg.sector_decomposition[:, slc]`` correspond to the
        :attr:`clock_symmetry`.

    TODO we may want to rename this class; 'ClockSite' should probably be the analogue to
         'SpinSite', so this should have an analogous name to 'SpinfulSite'
    """

    def __init__(self,
                 leg: ElementarySpace,
                 q: int,
                 clock_operators: np.ndarray,
                 clock_symmetry: ZNSymmetry | NoSymmetry,
                 clock_symmetry_sector_slice: slice,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None):
        self.q = q
        assert clock_operators.shape == (leg.dim, leg.dim, 2)
        self.clock_operators = clock_operators

        # check the clock_symmetry, and that the sectors of the leg come in correct multiplets
        self.clock_symmetry_sector_slice = slc = clock_symmetry_sector_slice
        assert leg.dim % q == 0
        if isinstance(clock_symmetry, ZNSymmetry):
            assert clock_symmetry.N == q
            expect = leg.num_sectors // q
            for i in range(q):
                assert np.sum(leg.sector_decomposition[:, slc] == i) == expect
        elif isinstance(clock_symmetry, NoSymmetry):
            pass
        else:
            raise TypeError('Invalid clock_symmetry')
        assert consistent_leg_symmetry(leg, clock_symmetry, slc)
        self.clock_symmetry = clock_symmetry

        Site.__init__(self, leg=leg, state_labels=state_labels, onsite_operators=onsite_operators)
        X, Z = [clock_operators[:, :, i] for i in range(2)]
        Xhc, Zhc = [np.conj(clock_operators[:, :, i].T) for i in range(2)]
        self.add_onsite_operator('Z', Z)
        self.add_onsite_operator('Zhc', Zhc)
        self.add_onsite_operator('Zphc', Z + Zhc)
        if isinstance(clock_symmetry, NoSymmetry):
            # TODO I (NK) am confused why Zphc was excluded from the Z_q symmetric operators in TeNPy v1?
            self.add_onsite_operator('X', X)
            self.add_onsite_operator('Xhc', Xhc)
            self.add_onsite_operator('Xphc', X + Xhc)

    def test_sanity(self):
        super().test_sanity()
        # check commutation relations
        X, Z = [self.clock_operators[:, :, i] for i in range(2)]
        Xhc, Zhc = [np.conj(self.clock_operators[:, :, i].T) for i in range(2)]
        assert np.allclose(X @ Z, np.exp(2.j * np.pi / self.q) * Z @ X)

        identity = np.eye(self.leg.num_sectors)
        assert np.allclose(np.linalg.matrix_power(X, self.q), identity)
        assert np.allclose(np.linalg.matrix_power(Z, self.q), identity)
        assert np.allclose(X @ Xhc, identity)
        assert np.allclose(Z @ Zhc, identity)

        # TODO check compatibility of the operators with the symmetry, i.e. eigenvalues
        #      match with charge sectors

    @classmethod
    def pure_clock_site(cls, q: int, conserve: Literal['Z_N', 'None'] = None):
        # TODO docs
        #      do we want to keep it like this? This has a different structure than for spins
        #      (SpinfulSite and SpinSite) and does not have a self.conserve, which SpinSite has

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

        state_labels = {str(n): n for n in range(q)}
        state_labels['up'] = 0
        if q % 2 == 0:
            state_labels['down'] = q // 2

        return ClockSite(leg=leg, q=q, clock_operators=clock_operators, clock_symmetry=sym,
                         clock_symmetry_sector_slice=slice(None, None), state_labels=state_labels)


class GoldenSite(Site):
    """TODO elaborate"""

    def __init__(self, handedness: Literal['left', 'right']):
        symmetry = FibonacciAnyonCategory(handedness=handedness)
        leg = ElementarySpace.from_basis(symmetry, [symmetry.vacuum, symmetry.tau])
        tau_occupation = SymmetricTensor.from_sector_projection([leg], symmetry.tau)
        Site.__init__(self, leg=leg, onsite_operators={'N': tau_occupation})


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
    if isinstance(leg.symmetry, ProductSymmetry):
        # symmetry_sector_slice tells us which factor in the leg symmetry is symmetry_factor
        slc = symmetry_sector_slice
        start = 0 if slc.start is None else slc.start
        stop = leg.symmetry.sector_slices[-1] if slc.stop is None else slc.stop
        factor_idx = np.searchsorted(leg.symmetry.sector_slices, start)
        return all([start == leg.symmetry.sector_slices[factor_idx],
                    stop == leg.symmetry.sector_slices[factor_idx + 1],
                    leg.symmetry.factors[factor_idx] == symmetry_factor])
    # TODO is this a reason to say that we should always work with ProductSymmetries?
    #      like tenpy v1, we always have a list of qmod, even if there is only 1
    return leg.symmetry == symmetry_factor


# TODO more sites:
#  - FermionSite (maybe name it SpinlessFermionSite for clarity?)
#  - SpinHalfFermionSite (or if its easy just do general spin?)
#  - SpinHalfHoleSite (i dont think this should inherit from FermionicSite, but not sure)
#  - BosonSite (maybe name it SpinlessBosonSite?)
#  - bosons with spin?
#  - what are relevant anyonic sites? already have Golden, but do some more
#  - remember to update cyten/__init__.py and cyten/models/__init__.py accordingly!
