from __future__ import annotations
import numpy as np
from typing import Literal, Sequence

from ..dtypes import Dtype
from ..backends import TensorBackend, get_backend
from ..backends.abstract_backend import Block
from ..spaces import ElementarySpace
from ..tensors import SymmetricTensor
from ..symmetries import (
    Symmetry, SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory,
    ProductSymmetry, BraidingStyle, FermionParity
)


# ==========================================================================================
# Design constraints / criteria
# ==========================================================================================
# FIXME rm
# factorized form:
#       eventually we need to extract an MPO-like form for multi-site operators
#       e.g. when building hamiltonian/WI/WII MPOs, or for correlation_function etc
# short identifiers:
#       We want a system, similar to tenpy v1, where we can refer to operators by a short and
#       intuitively readable (string?) identifier.
#       TBD: what is the scope/namespace where they must be unique?
#            per site-type? per model? globally in cyten?
#       TODO: really? or maybe not?
# customizing:
#       It should be (as) easy (as possible) to add custom sites / operators / models
# coupling different kinds of sites:
#       We will need the "same" coupling between many different kinds of sites.
#       E.g. Heisenberg between spin-1/2 sites, between spin-1 sites, or between fermions, or
#       between mixed types.
#       This is easy in tenpy v1, since local operators are referred to by name only, and all
#       of those site types define e.g. Sz, Sp, Sm.
#       We should preserve that and have an interface to add a heisenberg coupling that is not
#       specialized to the site types, and e.g. still works unmodified if a site is changed.
# allow MPOGraph optimizations
#       The framework should allow MPOGraph to do its optimization, though the graph may
#       need to change. This may require us to deal with a basis for elementary local MPO parts.
#       Elementary meaning their virutal legs only have a single sector, since these "parts" are
#       the edges in the MPO graph. Problem: there may be an infinite number of possible parts,
#       if there is an infinite number of sectors, as e.g. in U(1).
# uses: Hamiltonian MPO, WII MPO, expval, corr-func
# debugging: verify a Model; pretty print the model you defined
# do we support algebra, i.e. do couplings know they are sums of others



class Site:
    """Collects necessary information about a single local site of a lattice model.

    A site defines the the local Hilbert space in terms of its :attr:`leg`.
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
        The local physical Hilbert space
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
            # FIXME backend, device?
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


# TODO how do we deal with grouped site? -> FIXME new issue?
# TODO set common charges  # FIXME new issue (there should be code somewhere)


class SpinfulSite(Site):
    """Common base class for sites that have a spin degree of freedom.

    TODO find a good format to doc the onsite operators that exist in a site


    Attributes
    ----------
    double_total_spin : int
        Twice the :attr:`total_spin`. We store this, because it is an integer.
    spin_vector : 3D array
        The vector of spin operators as a numpy array with axes ``[p, p*, i]`` and shape
        ``(dim, dim, 3)``.
    spin_symmetry : SU2Symmetry | U1Symmetry | ZNSymmetry | NoSymmetry
        The symmetry of the spin degree of freedom that is enforced.
        We can conserve::

            - SU(2), the full spin rotation symmetry
            - U(1), with conserved charge ``2 * Sz``
            - Z_2, with conserved charge ``(Sz + S_tot) % 2``.
            - nothing

        The full :attr:`symmetry` must either coincide with the `spin_symmetry`, or it must be
        a :class:`ProductSymmetry` with the `spin_symmetry` as a factor.
    spin_symmetry_sector_slice : slice
        A slice such that the entries ``leg.sector_decomposition[:, slc]`` correspond to the
        :attr:`spin_symmetry`.
    """
    def __init__(self,
                 leg: ElementarySpace,
                 total_spin: float,
                 spin_vector: np.ndarray,
                 spin_symmetry: SU2Symmetry | U1Symmetry | ZNSymmetry | NoSymmetry,
                 spin_symmetry_sector_slice: slice,
                 state_labels: dict[str, int] = None,
                 onsite_operators: dict[str, SymmetricTensor] = None):
        self.double_total_spin = twoS = int(round(2 * total_spin, 0))
        if twoS >= 0:
            raise ValueError('Negative spin.')
        if np.allclose(twoS / 2, total_spin):
            raise ValueError('total_spin must be half integer: 0, 1/2, 1, 3/2, ...')
        assert spin_vector.shape == (leg.dim, leg.dim, 3)
        self.spin_vector = spin_vector

        # check the spin_symmetry, and that the sectors of the leg come in correct multiplets
        self.spin_symmetry_sector_slice = slc = spin_symmetry_sector_slice
        assert leg.dim % (twoS + 1) == 0
        if isinstance(spin_symmetry, SU2Symmetry):
            # all must be in the same single spin sector
            assert np.all(leg.sector_decomposition[:, slc] == twoS)
        elif isinstance(spin_symmetry, U1Symmetry):
            # make sure every Sz sector appears the same number of times.
            expect = leg.num_sectors // (twoS + 1)
            for m in range(-twoS, twoS + 2, 2):
                num_sectors = np.sum(leg.sector_decomposition[:, slc] == twoS)
                assert num_sectors == expect
        elif isinstance(spin_symmetry, ZNSymmetry):
            assert spin_symmetry.N == 2
            num_m = twoS + 1
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
        if isinstance(leg.symmetry, ProductSymmetry):
            assert any(factor == spin_symmetry for factor in leg.symmetry.factors)
        else:
            # TODO is this a reason to say that we should always work with ProductSymmetries?
            #      like tenpy v1, we always have a list of qmod, even if there is only 1
            assert leg.symmetry == spin_symmetry
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
        eigenvalue = self.double_total_spin * (self.double_total_spin + 2) // 4
        assert np.allclose(S_sq, eigenvalue * np.eye(self.double_total_spin + 1))

    @property
    def total_spin(self) -> float:
        return self.double_total_spin / 2


class FermionicSite(Site):
    """TODO similar structure and role as SpinfulSite

    Mutually exclusive with BosonicSite, but compatible with SpinfulSite

    TODO need to implement FermionOccupation(Symmetry)
         sectors and fusion like U(1)
         braiding similar to FermionParity: -1 if both sectors are odd and +1 otherwise

    Can conserve:
        - each individual species occupation N_i -> ProductSymmetry of k FermionOccupation
        - per-species parity ``N_i % 2`` -> ProductSymmetry of k FermionParity
        - total fermion number sum_i N_i
        - fermion parity (sum_i N_i) % 2

    """
    creators: np.ndarray  # [p, p*, i] where i are different species ;  == [Cd0, Cd1, ...]
    annihilators: np.ndarray  # [p, p*, i] ;  == [C0, C1, ...]
    occupation_symmetry: ProductSymmetry | FermionOccupation | FermionParity


class BosonicSite(Site):
    """TODO similar to FermionicSite, but with bosons."""
    creators: np.ndarray  # [p, p*, i] where i are different species ;  == [Bd0, Bd1, ...]
    annihilators: np.ndarray  # [p, p*, i] ;  == [B0, B1, ...]
    occupation_symmetry: ProductSymmetry | U1Symmetry | ZNSymmetry | NoSymmetry


# Concrete classes::
# ------------------------------------------------------------------------------------------


class SpinSite(SpinfulSite):
    spin: float = .5


class GoldenSite(Site):
    def __init__(self, handedness):
        symmetry = FibonacciAnyonCategory(handedness)
        leg = ElementarySpace.from_basis(symmetry, [symmetry.vacuum, symmetry.tau])
        tau_occupation = SymmetricTensor.from_sector_projection([leg], symmetry.tau)
        Site.__init__(self, leg=leg, onsite_operators={'N': tau_occupation})


# ==========================================================================================
# Couplings : multi-site operators that are used to build Hamiltonian MPOs,
#             can measure correlation functions etc.
# ==========================================================================================


class Coupling:
    name: str
    sites: list[Site]
    factorization: list[SymmetricTensor]  # [vL, p, vR, p*]

    # FIXME should this have add and mul and so on?

    @classmethod
    def from_numpy(self, operator: np.ndarray, sites: list[Site]):
        ...

    @classmethod
    def from_tensor(self, operator: SymmetricTensor, sites: list[Site]):
        ...

    def to_tensor(self) -> SymmetricTensor:
        ...

    def to_numpy(self) -> np.ndarray:
        return self.to_tensor().to_numpy()


class OnSiteOperator(Coupling):
    operator: SymmetricTensor  # [p, p*]


# ==========================================================================================
# Mockup model: how do we generate the concrete coupling instance in practice?
# ==========================================================================================

class CouplingModel:

    # FIXME should have a dict[str, Coupling] and/or dict[str, CouplingMaker]

    def add_coupling(self, prefactor, u1, u2, dx, coupling):
        # TODO need to adjust this as well...
        ...

    def get_coupling(self, name: str, u1, u2, dx):
        # FIXME lookup, deal with attr vs meth differently
        # FIXME eg. MPS.expval takes Couplings and Simulation can use the model to 
        #       tranlate a string name to a Coupling
        ...

    ...


class HeisenbergModel(CouplingModel):

    def init_sites(self):
        ...
        #  FIXME make dict of couplings

    # TODO code to create couplings is convenient if it lives in the model, so we can access it
    #      also for expval / correlation_function, but it could also live elsewhere
    def heisenberg_coupling(self, site1: SpinfulSite, site2: SpinfulSite):
        # TODO caching?
        h = np.dot(site1.spin_vector, site2.spin_vector, (0, 0))  # [p0, p0*, p1, p1*]
        h = np.transpose(h, [0, 2, 1, 3])  # [p0, p1, p0*, p1*]
        return Coupling.from_numpy(h, [site1, site2])

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1., 'real_or_array'))
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # TODO maybe we need to rework how we pass the spatial information around.
            #      for now, I am forcing the new Coupling class into this framework...
            # TODO adjust the convenience methods in CouplingModel (add_coupling and friends)
            #      to meet this code halfway.
            h = self.heisenberg_coupling(self.lat.unit_cell[u1], self.lat.unit_cell[u2])
            self.add_coupling(-J, u1, u2, dx, h)


class GoldenChain(CouplingModel):

    def golden_coupling(self, site1: Site, site2: Site):
        # TODO generalize to additional factors in the symmetry?
        assert isinstance(site1.symmetry, FibonacciAnyonCategory)
        assert isinstance(site2.symmetry, FibonacciAnyonCategory)
        h = SymmetricTensor.from_sector_projection([site1.leg, site2.leg], sector=[1])  # TODO backend?
        return Coupling.from_tensor(h, [site1, site2])

    def add_golden_coupling(self, prefactor, u1, u2, dx):
        h = self.golden_coupling(self.lat.unit_cell[u1], self.lat.unit_cell[u2])
        self.add_coupling(prefactor, u1, u2, dx, h)

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1., 'real_or_array'))
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_golden_coupling(-J, u1, u2, dx)
