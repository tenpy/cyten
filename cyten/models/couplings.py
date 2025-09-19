"""Couplings are the building blocks of Hamiltonians for lattice models.

This module defines a base class for couplings, which are given in a MPO-like factorized form,
as well as functions that create common couplings such as e.g. a Heisenberg couplings between
two sites that have a spin degree of freedom.
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np

from ..symmetries import FibonacciAnyonCategory, Sector
from ..dtypes import Dtype
from ..backends.abstract_backend import Block
from ..tensors import (
    SymmetricTensor, squeeze_legs, tdot, add_trivial_leg, permute_legs, svd, scale_axis, compose,
    outer
)
from .degrees_of_freedom import (
    DegreeOfFreedom, SpinDOF, BosonicDOF, FermionicDOF, ClockDOF, get_same_DOF_backend,
    get_same_DOF_device
)
from .sites import GoldenSite


class Coupling:
    """A coupling is a (usually hermitian) operator on a few :class:`Site` s.

    TODO elaborate, use case, examples, maybe look at the docs in tenpy v1 of terms or add_coupling

    Attributes
    ----------
    sites : list of :class:`Site`
        The sites that the operators act on.
    factorization : list of :class:`SymmetricTensor`
        A list of tensors that, if contracted, give the operator that is represented.
        Each tensor ``factorization[i]`` has legs ``[wL, pi, wR, pi*]``, where ``pi`` and ``pi*``
        are the physical :attr:`Site.leg` of the corresponding ``sites[i]``, and where contracting
        the ``wL`` and ``wR`` legs in an MPO-like geometry gives the multi-site operator.
        TODO should we rename vL/vR to wL/wR to match tenpy MPO convention?
        TODO do we want to keep the convention ``[wL, p, wR, p*]`` or maybe change it to
             ``[wL, pi, wR, pi*]``? Then it would be consistent with the labels of the input
             tensors in `from_tensor`.
    name : str, optional
        A descriptive name that can be used when pretty-printing, to identify the coupling.
        For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
    """

    def __init__(self, sites: list[DegreeOfFreedom], factorization: list[SymmetricTensor],
                 name: str = None):
        self.sites = sites
        assert len(factorization) == len(sites)
        self.factorization = factorization
        self.name = name
        self.test_sanity()  # OPTIMIZE

    def test_sanity(self):
        backend = get_same_DOF_backend(*self.sites)
        device = get_same_DOF_device(*self.sites)
        for i, (s, W) in enumerate(zip(self.sites, self.factorization)):
            s.test_sanity()
            W.test_sanity()
            assert W.backend == backend
            assert W.device == device
            assert W.num_codomain_legs == 2
            assert W.num_domain_legs == 2
            assert W.labels == ['wL', f'p{i}', 'wR', f'p{i}*']
            assert W.get_leg_co_domain(f'p{i}') == s.leg
            assert W.get_leg_co_domain(f'p{i}*') == s.leg
        assert self.factorization[0].get_leg('wL').is_trivial
        for W1, W2 in zip(self.factorization[:-1], self.factorization[1:]):
            assert W1.get_leg_co_domain('wR') == W2.get_leg_co_domain('wL')
        assert self.factorization[-1].get_leg('wR').is_trivial

    @classmethod
    def from_dense_block(cls, operator: Block, sites: list[DegreeOfFreedom], name: str = None,
                         dtype: Dtype = None) -> Coupling:
        """Convert a dense block to a :class:`Coupling`.

        Parameters
        ----------
        block : Block
            The data to be converted to a Coupling as a backend-specific block or some data that
            can be converted using :meth:`BlockBackend.as_block`. The order of axes must match the
            `sites`, that is, the axes correspond to ``[p0, p1, ..., p1*, p0*]`` (codomain legs
            ascending, domain legs descending), where ``pi`` corresponds to site ``sites[i]``.
            The block should be given in the "public" basis order of the sites, i.e.,
            according to `sites[i].sectors_of_basis`.
        sites : list of :class:`Site`
            The sites that the operators act on.
        name : str, optional
            A descriptive name that can be used when pretty-printing, to identify the coupling.
        backend : :class:`TensorBackend`, optional
            If given, the backend of the tensors in the factorization. Per default, the default
            backend compatible with the symmetry.
        device : str, optional
            If given, the block is moved to that device. Per default, try to use the device of
            the `block`, if it is a backend-specific block, or fall back to the backends default
            device.
        dtype : :class:`Dtype`, optional
            If given, the block is converted to that dtype and the resulting tensors in the
            factorization will have that dtype. By default, we detect the dtype from the block.
        """
        backend = get_same_DOF_backend(*sites)
        device = get_same_DOF_device(*sites)
        co_domain = [s.leg for s in sites]
        p_labels = [f'p{i}' for i in range(len(sites))]
        labels = [*p_labels, *[f'{pi}*' for pi in p_labels][::-1]]
        op = SymmetricTensor.from_dense_block(operator, co_domain, co_domain, backend=backend,
                                              labels=labels, dtype=dtype, device=device)
        return cls.from_tensor(op, sites=sites, name=name)

    @classmethod
    def from_tensor(cls, operator: SymmetricTensor, sites: list[DegreeOfFreedom], name: str = None
                    ) -> Coupling | OnSiteOperator:
        """Convert an operator / tensor to a :class:Coupling.
        
        Decompose a (in general multi-site) operator into factors using SVD such that contracting
        the factors again reproduces the operator.

        .. note ::
            For symmetries with non-symmetric braids, the decomposition depends on the levels of
            the legs that determine whether over-braids or under-braids occur when exchanging legs.
            The convention here is to assign higher levels to legs "further to the right", i.e.,
            the legs corresponding to the labels `p2` and `p2*` have higher levels than `p1` and
            `p1*`, and lower levels than `p3` and `p3*`.

        Parameters
        ----------
        operator : :class:`SymmetricTensor`
            Operator to be converted to a coupling. The legs should be ordered as
            ``[p0, p1, ..., p1*, p0*]``, where ``pi`` and ``pi*`` correspond to the legs associated
            with site ``sites[i]``.
        sites : list of :class:`Site`
            The sites that the operator acts on.
        name : str, optional
            A descriptive name that can be used when pretty-printing, to identify the coupling.
            For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
        """
        assert operator.backend == get_same_DOF_backend(*sites)
        assert operator.device == get_same_DOF_device(*sites)
        assert operator.codomain.factors == [site.leg for site in sites]
        assert operator.domain.factors == operator.codomain.factors

        if len(sites) == 1:
            return OnSiteOperator.from_tensor(operator, sites, name=name)

        # decompose from right to left using SVD
        # convention for levels: legs to the right have higher levels
        factorization = []
        n = operator.num_codomain_legs
        levels = [2 * i for i in range(n)]
        levels.extend([2 * i + 1 for i in range(n)][::-1])

        U = permute_legs(operator, domain=[n, n - 1], levels=levels, bend_right=True)
        U, S, V = svd(U, ['wR', 'wL'])
        U = scale_axis(U, S, leg='wR')
        V = permute_legs(V, codomain=[0, 1], bend_right=True)
        V = add_trivial_leg(V, domain_pos=1, label='wR')
        factorization.append(V)

        for n in range(operator.num_codomain_legs - 1, 1, -1):
            levels = [2 * i for i in range(n)]
            levels.extend([2 * i + 1 for i in range(n)][::-1])
            # for the leg connecting to the previous operator that is already in factorization
            levels.append(2 * n)
            U = permute_legs(U, domain=[n, -1, n - 1], levels=levels, bend_right=True)
            # the legs are now [p0, p1, ..., p{n-2}, p{n-2}*, p1*, p0*, p{n-1}, wR, p{n-1}*]
            U, S, V = svd(U, ['wR', 'wL'])
            U = scale_axis(U, S, leg='wR')
            V = permute_legs(V, codomain=[0, 1], bend_right=True)
            factorization.append(V)

        U = permute_legs(U, domain=[1, 2], levels=[0, 1, 2], bend_right=True)
        U = add_trivial_leg(U, codomain_pos=0, label='wL')
        factorization.append(U)
        factorization = factorization[::-1]
        return Coupling(sites=sites, factorization=factorization, name=name)

    def to_tensor(self) -> SymmetricTensor:
        """Convert to a tensor."""
        # the convention for the decomposition is that legs to the right have
        # higher levels -> this must now also hold
        res = squeeze_legs(self.factorization[0], 'wL')
        for n, W in enumerate(self.factorization[1:]):
            levels = list(range(2 * n + 1))
            levels.extend([2 * n + 2, 2 * n + 1])
            # legs are (before permute) [p0, p0*, p1, p1*, ..., pn, wR, pn*]
            res = permute_legs(res, domain=['wR'], levels=levels, bend_right=True)
            W_ = permute_legs(W, codomain=['wL'], bend_right=True)
            res = tdot(res, W_, 'wR', 'wL')
        res = squeeze_legs(res, 'wR')
        codom_labels = [f'p{i}' for i in range(len(self.sites))]
        dom_labels = [l + '*' for l in codom_labels]
        # legs are now [p0, p0*, p1, p1*, ...]
        res = permute_legs(res, codomain=codom_labels, domain=dom_labels,
                           levels=list(range(2 * len(self.sites))), bend_right=True)
        return res

    def to_numpy(self) -> np.ndarray:
        """Convert to a numpy array."""
        return self.to_tensor().to_numpy()


class OnSiteOperator(Coupling):
    """A (usually hermitian) on-site operator acting on a single :class:`Site`.

    Similar to :class:`Coupling`, but must act on one :class:`Site`.

    TODO examples

    Attributes
    ----------
    operator : :class:`SymmetricTensor`
        Tensor representing the on-site operator with legs ``[p, p*]``, where ``p`` and ``p*`` are
        the physical space :attr:`sites[0].leg`.
    sites : list of :class:`Site`
        Contains the single site that `operator` acts on.
    factorization : list of :class:`SymmetricTensor`
        Contains a single tensor corresponding to `operator` with added trivial legs for `wL` and
        `wR`.
    name : str, optional
        A descriptive name that can be used when pretty-printing, to identify the coupling.
        For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.

    See Also
    --------
    :class:`Coupling`
    """

    def __init__(self, site: DegreeOfFreedom, operator: SymmetricTensor, name: str = None):
        self.operator = operator
        W = add_trivial_leg(operator, codomain_pos=0, label='wL')
        W = add_trivial_leg(W, domain_pos=1, label='wR')
        # TODO do this for now; discuss how we actually want to do it
        W.relabel({'p': 'p0', 'p*': 'p0*'})
        Coupling.__init__(self, sites=[site], factorization=[W], name=name)

    @classmethod
    def from_tensor(cls, operator: SymmetricTensor, sites: list[DegreeOfFreedom], name: str = None
                    ) -> OnSiteOperator:
        assert len(sites) == 1
        return cls(site=sites[0], operator=operator, name=name)


# SPIN COUPLINGS

def spin_spin_coupling(sites: list[SpinDOF], xx: float = None, yy: float = None,
                       zz: float = None, name: str = 'spin-spin') -> Coupling:
    """Coupling between two spins.

    Parameters
    ----------
    xx, yy, zz : float, optional
        If given, adds a corresponding term, e.g. ``xx * S_i^x S_j^x`` with the value as prefactor.
    TODO do we need mixed combinations, like S_i^x S_j^y ?
    """
    # TODO test that this builds what we expect
    assert len(sites) == 2
    s1 = sites[0].spin_vector
    s2 = sites[1].spin_vector
    h = 0  # build in leg order [p0, p0*, p1, p1*] and transpose only once before returning
    if xx is not None:
        h += xx * np.tensordot(s1[:, :, 0], s2[:, :, 0], axes=0)
    if yy is not None:
        h += yy * np.tensordot(s1[:, :, 1], s2[:, :, 1], axes=0)
    if zz is not None:
        h += zz * np.tensordot(s1[:, :, 2], s2[:, :, 2], axes=0)
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    h = np.transpose(h, [0, 2, 3, 1])
    return Coupling.from_dense_block(h, sites, name=name)


def spin_field_coupling(sites: list[SpinDOF], hx: float = None, hy: float = None,
                        hz: float = None, name: str = 'spin-field') -> Coupling:
    """Coupling between a (single) spin and a (magnetic) field.

    Parameters
    ----------
    hx, hy, hz : float, optional
        If given, adds a corresponding term ``hx * S_i^x``, ``hy * S_i^y``, ``hz * S_i^z`` with
        the value as prefactor.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 1
    s = sites[0].spin_vector
    h = 0
    if hx is not None:
        h += hx * s[:, :, 0]
    if hy is not None:
        h += hy * s[:, :, 1]
    if hz is not None:
        h += hz * s[:, :, 2]
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    return Coupling.from_dense_block(h, sites, name=name)


def aklt_coupling(sites: list[SpinDOF], name: str = 'AKLT') -> Coupling:
    r"""AKLT coupling between two spin-1 sites.
    
    Construct the AKLT coupling between S=1 spins as originally defined by
    Affleck, Kennedy, Lieb, Tasaki in :cite:`affleck1987`, but drop the
    constant part of 1/3 per bond and rescale with a factor of 2.

    .. math ::
        2 * P^{S=2}_{i,i+1} + const
        = \vec{S}_i \cdot \vec{S}_{i+1}
          + \frac{1}{3} (\vec{S}_i \cdot \vec{S}_{i+1})^2
    """
    # TODO test that this builds what we expect
    assert len(sites) == 2
    assert sites[0].double_total_spin == 2 == sites[1].double_total_spin
    s1 = sites[0].spin_vector
    s2 = sites[1].spin_vector
    S_dot_S = np.tensordot(s1, s2, axes=[2, 2])
    S_dot_S = np.transpose(S_dot_S, [0, 2, 3, 1])
    S_dot_S_square = np.tensordot(S_dot_S, S_dot_S, axes=[[3, 2], [0, 1]])
    h = S_dot_S + S_dot_S_square / 3.
    return Coupling.from_dense_block(h, sites, name=name)


def heisenberg_coupling(sites: list[SpinDOF], name: str = 'S.S') -> Coupling:
    # TODO test that this builds what we expect
    return spin_spin_coupling(sites=sites, xx=1, yy=1, zz=1, name=name)


def chiral_3spin_coupling(sites: list[SpinDOF], name: str = 'S.SxS') -> Coupling:
    # TODO test that this builds what we expect
    assert len(sites) == 3
    SxS = np.cross(sites[1].spin_vector[:, None, None, :, :],
                   sites[2].spin_vector[None, :, :, None, :],
                   axis=4)  # [p1, p2, p2*, p1*, i]
    h = np.tensordot(sites[0].spin_vector, SxS, (-1, -1))  # [p0, p0*, p1, p2, p2*, p1*]
    h = np.transpose(h, [0, 2, 3, 4, 5, 1])
    return Coupling.from_dense_block(h, sites, name=name)


# BOSON AND FERMION COUPLINGS

def chemical_potential(sites: list[BosonicDOF | FermionicDOF], mu: float | list[float | None] = None,
                       name: str = 'chem. pot.') -> Coupling:
    """Chemical potential for bosons or fermions on a single site.

    Parameters
    ----------
    mu: float | list[float | None]
        If given, adds the corresponding term ``-1 * mu[j] n_i(j)`` with the value as prefactor,
        where `j` refers to the boson or fermion species. If there are multiple species and `mu`
        is a `float`, it is applied as chemical potential to all particle species.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 1
    site = sites[0]
    h = None
    if isinstance(mu, (list, np.ndarray)):
        msg = f'Invalid number of entries in `mu`: {len(mu)} != {site.num_species}'
        assert len(mu) == site.num_species, msg
        first_contri = True
        for i, mu_i in enumerate(mu):
            if mu_i is None:
                continue
            op_name = 'N' if site.num_species == 1 else f'N{i}'
            if first_contri:
                first_contri = False
                h = -1 * mu_i * site.onsite_operators[op_name]
                continue
            h -= mu_i * site.onsite_operators[op_name]
    else:
        op_name = 'N' if site.num_species == 1 else 'Ntot'
        h = -1 * mu * site.onsite_operators[op_name]
    if h is None:
        raise ValueError('Must have at least one non-zero prefactor.')
    return Coupling.from_tensor(h, sites=sites, name=name)


def onsite_interaction(sites: list[BosonicDOF | FermionicDOF], name: str = 'onsite interaction'
                       ) -> Coupling:
    """Onsite interactions for bosons or fermions.

    Corresponds to ``n_i**2 / 2``, where `n_i` is the total onsite particle
    number (including all particle species).
    """
    # TODO test that this builds what we expect
    assert len(sites) == 1
    op_name = 'N' if sites[0].num_species == 1 else 'Ntot'
    h = sites[0].onsite_operators[op_name]
    h = compose(h, h) / 2
    return Coupling.from_tensor(h, sites=sites, name=name)


def long_range_interaction(sites: list[BosonicDOF | FermionicDOF], distance: int,
                           name: str = 'long-range interaction') -> Coupling:
    """Long-range density-density interactions for bosons or fermions.

    Corresponds to ``n_i n_j``, where `n_i` and `n_j` are the total onsite
    particle numbers (including all particle species) of the sites `i` and `j`
    with a given distance between them, that is, `j == i + distance`.
    """
    # TODO test that this builds what we expect
    assert isinstance(distance, int)
    # TODO should distance == 0 give the onsite case?
    assert distance > 0, 'distance between the sites must be positive'
    assert len(sites) == distance + 1
    sites_ij = [sites[0], sites[-1]]
    op_names = ['N' if site.num_species == 1 else 'Ntot' for site in sites_ij]
    ops = [site.onsite_operators[op_name] for site, op_name in zip(sites_ij, op_names)]
    h = ops[0]
    for i in range(1, distance):
        identity = SymmetricTensor.from_eye([sites[i].leg], labels=[f'p{i}', f'p{i}*'])
        h = outer(h, identity)
    h = outer(h, ops[1], relabel1={'p': 'p0', 'p*': 'p0*'},
              relabel2={'p': f'p{distance}', 'p*': f'p{distance}*'})
    return Coupling.from_tensor(h, sites=sites, name=name)


def nearest_neighbor_interaction(sites: list[BosonicDOF | FermionicDOF],
                                 name: str = 'NN interaction') -> Coupling:
    """Nearest neighbor interactions for bosons or fermions.

    Corresponds to ``n_i n_j``, where `n_i` and `n_j` are the total onsite
    particle numbers (including all particle species) of the two neighboring
    sites.
    """
    # TODO test that this builds what we expect
    return long_range_interaction(sites=sites, distance=1, name=name)


def next_nearest_neighbor_interaction(sites: list[BosonicDOF | FermionicDOF],
                                      name: str = 'NNN interaction') -> Coupling:
    """Next nearest neighbor interactions for bosons or fermions.

    Corresponds to ``n_i n_j``, where `n_i` and `n_j` are the total onsite
    particle numbers (including all particle species) of the two next nearest
    neighboring sites.
    """
    # TODO test that this builds what we expect
    return long_range_interaction(sites=sites, distance=2, name=name)


# CLOCK COUPLINGS

def clock_clock_coupling(sites: list[ClockDOF], xx: float = None, zz: float = None,
                         name: str = 'clock-clock') -> Coupling:
    r"""Coupling between two clock sites.

    Parameters
    ----------
    xx, zz : float, optional
        If given, adds the corresponding term, ``xx * (X_i X_j^\dagger + \mathrm{ h.c.})``, and
        ``zz * (Z_i Z_j^\dagger + \mathrm{ h.c.})``, with the value as prefactor.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 2
    clock1 = sites[0].clock_operators
    clock2 = sites[1].clock_operators
    h = 0
    if xx is not None:
        h += xx * np.tensordot(clock1[:, :, 0], np.conj(clock2[:, :, 0].T), axes=0)
        h += xx * np.tensordot(np.conj(clock1[:, :, 0].T), clock2[:, :, 0], axes=0)
    if zz is not None:
        h += zz * np.tensordot(clock1[:, :, 1], np.conj(clock2[:, :, 1].T), axes=0)
        h += zz * np.tensordot(np.conj(clock1[:, :, 1].T), clock2[:, :, 1], axes=0)
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    h = np.transpose(h, [0, 2, 3, 1])
    return Coupling.from_dense_block(h, sites, name=name)


def clock_field_coupling(sites: list[ClockDOF], hx: float = None, hz: float = None,
                         name: str = 'clock-field') -> Coupling:
    r"""Coupling between a clock site and a (magnetic) field.

    Parameters
    ----------
    hx, hz: float, optional
        If given, adds the corresponding term, ``hx * (X_i + \mathrm{ h.c.})``, and
        ``hz * (Z_i + \mathrm{ h.c.})``, with the value as prefactor.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 1
    clock = sites[0].clock_operators
    h = 0
    if hx is not None:
        Xphc = clock[:, :, 0] + np.conj(clock[:, :, 0].T)
        h += hx * Xphc
    if hz is not None:
        Zphc = clock[:, :, 1] + np.conj(clock[:, :, 1].T)
        h += hz * Zphc
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    return Coupling.from_dense_block(h, sites, name=name)


# ANYONIC COUPLINGS

def multi_site_projector(sites: list[DegreeOfFreedom], sector: Sector, name: str) -> Coupling:
    """Coupling between multiple sites that corresponds to a projector onto a common sector."""
    backend = get_same_DOF_backend(*sites)
    device = get_same_DOF_device(*sites)
    labels = [f'p{i}' for i in range(len(sites))]
    labels = [*labels, *[f'{l}*' for l in reversed(labels)]]
    # TODO complex dtype is needed here for anyonic symmetries;
    # we should maybe automatically do this in the tensor class and not fix it here
    projector = SymmetricTensor.from_sector_projection(
        [s.leg for s in sites], sector=sector, backend=backend, labels=labels, device=device,
        dtype=Dtype.complex128
    )
    return Coupling.from_tensor(projector, sites=sites, name=name)


def two_site_projector(sites: list[DegreeOfFreedom], sector: Sector, name: str) -> Coupling:
    """Coupling between two sites that corresponds to a projector onto a common sector."""
    assert len(sites) == 2
    return multi_site_projector(sites=sites, sector=sector, name=name)


def three_site_projector(sites: list[DegreeOfFreedom], sector: Sector, name: str) -> Coupling:
    """Coupling between three sites that corresponds to a projector onto a common sector."""
    assert len(sites) == 3
    return multi_site_projector(sites=sites, sector=sector, name=name)


def gold_coupling(sites: list[GoldenSite], name: str = 'P_vac') -> Coupling:
    """Coupling between two Fibonacci anyons projecting onto their trivial channel."""
    for site in sites:
        assert isinstance(site.symmetry, FibonacciAnyonCategory)
        assert site.leg.sector_decomposition_where(np.array([1])) is not None
    return two_site_projector(sites, sector=np.array([0]), name=name)

# TODO implement more of these functions that generate couplings. at least cover everything
#      that occurs as a coupling in tenpy v1. also cover some anyon models.
#      if it is a "common" coupling, put it in cyten, if it is "obscure" or very specific,
#      put it in tenpy (for now, we keep a tenpy_models mockup in cyten)
# TODO update cyten/__init__.py and cyten/models/__init__.py accordingly!
