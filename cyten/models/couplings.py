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
from ..backends.abstract_backend import Block, get_same_backend
from ..tensors import (
    SymmetricTensor, squeeze_legs, add_trivial_leg, permute_legs, compose, outer,
    horizontal_factorization
)
from .degrees_of_freedom import DegreeOfFreedom, SpinDOF, BosonicDOF, FermionicDOF, ClockDOF
from .sites import GoldenSite


class Coupling:
    """A coupling is an operator on a few :class:`Site` s, factorized as one tensor per site.

    The intended use case is to build tensor network representations (e.g. MPOs) of Hamiltonians.

    Attributes
    ----------
    sites : list of :class:`Site`
        The sites that the operators act on.
    factorization : list of :class:`SymmetricTensor`
        A list of tensors that, if contracted, give the operator that is represented.
        Each tensor ``factorization[i]`` has legs ``[wL, pi, wR, pi*]``, where ``pi`` and ``pi*``
        are the physical :attr:`Site.leg` of the corresponding ``sites[i]``, and where contracting
        the ``wL`` and ``wR`` legs in an MPO-like geometry gives the multi-site operator.
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
        backend = get_same_backend(*self.sites)
        for i, (s, W) in enumerate(zip(self.sites, self.factorization)):
            s.test_sanity()
            W.test_sanity()
            assert W.backend == backend
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
                         dtype: Dtype = None, understood_braiding: bool = False) -> Coupling:
        """Convert a dense block to a :class:`Coupling`.

        Parameters
        ----------
        operator : Block
            The data to be converted to a Coupling as a backend-specific block or some data that
            can be converted using :meth:`BlockBackend.as_block`. The order of axes must match the
            `sites`, that is, the axes correspond to ``[p0, p1, ..., p1*, p0*]`` (codomain legs
            ascending, domain legs descending), where ``pi`` corresponds to site ``sites[i]``.
            The block should be given in the "public" basis order of the sites, i.e.,
            according to ``sites[i].sectors_of_basis``.
        sites : list of :class:`Site`
            The sites that the operators act on.
        name : str, optional
            A descriptive name that can be used when pretty-printing, to identify the coupling.
        dtype : :class:`Dtype`, optional
            If given, the block is converted to that dtype and the resulting tensors in the
            factorization will have that dtype. By default, we detect the dtype from the block.
        """
        backend = get_same_backend(*sites)
        device = sites[0].default_device
        assert all(s.default_device == device for s in sites[1:])
        co_domain = [s.leg for s in sites]
        p_labels = [f'p{i}' for i in range(len(sites))]
        labels = [*p_labels, *[f'{pi}*' for pi in p_labels][::-1]]
        op = SymmetricTensor.from_dense_block(operator, co_domain, co_domain, backend=backend,
                                              labels=labels, dtype=dtype, device=device,
                                              understood_braiding=understood_braiding)
        return cls.from_tensor(op, sites=sites, name=name)

    @classmethod
    def from_tensor(cls, operator: SymmetricTensor, sites: list[DegreeOfFreedom], name: str = None,
                    cutoff_singular_values: float = None,
                    ) -> Coupling:
        """Convert an operator / tensor to a :class:Coupling.

        Decomposes an operator into factors using :func:`cyten.horizontal_factorization` to
        obtain the :attr:`factorization` of the coupling.

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
        cutoff_singular_values : float, optional
            If given, truncate singular values (see :func:`cyten.horizontal_factorization`)
            below this threshold.
        """
        assert operator.backend == get_same_backend(*sites)
        assert operator.codomain.factors == [site.leg for site in sites]
        assert operator.domain.factors == operator.codomain.factors

        W, rest = horizontal_factorization(operator, 1, 1, new_labels=['wR', 'wL'],
                                           cutoff_singular_values=cutoff_singular_values)
        factorization = [add_trivial_leg(W, codomain_pos=0, label='wL')]
        for n in range(len(sites) - 2):
            W, rest = horizontal_factorization(rest, 1, 1, new_labels=['wL', 'wR'],
                                               cutoff_singular_values=cutoff_singular_values)
            factorization.append(W)
        assert (rest.num_codomain_legs, rest.num_domain_legs) == (2, 1)
        factorization.append(add_trivial_leg(rest, domain_pos=1, label='wR'))
        return Coupling(sites=sites, factorization=factorization, name=name)

    @property
    def num_sites(self):
        return len(self.sites)

    def to_tensor(self) -> SymmetricTensor:
        """Convert to a tensor."""
        # TODO : this would be a great use case for a planar diagram as well...
        res = squeeze_legs(self.factorization[0], 'wL')
        res = permute_legs(res, [-1, 0], [1], bend_right=False)
        for n in range(1, self.num_sites):
            W = permute_legs(self.factorization[n], ['wL'], [f'p{n}*', 'wR', f'p{n}'],
                             bend_right=True)
            res = compose(res, W)
            res = permute_legs(res, [-1, *range(2 * n), 2 * n], [-2],
                               bend_right={-1: False, -3: True})
        res = squeeze_legs(res, 'wR')
        codom_labels = [f'p{i}' for i in range(len(self.sites))]
        dom_labels = [l + '*' for l in codom_labels]
        res = permute_legs(res, codom_labels, dom_labels, bend_right=False)
        return res

    def to_numpy(self) -> np.ndarray:
        """Convert to a numpy array."""
        return self.to_tensor().to_numpy()


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

def chemical_potential(sites: list[BosonicDOF | FermionicDOF], mu: float | list[float | None] = 1.,
                       name: str = 'chem. pot.') -> Coupling:
    """Chemical potential for bosons or fermions on a single site.

    Parameters
    ----------
    mu : float | list[float | None]
        Chemical potential. Add the corresponding term ``-1 * mu[j] n_i(j)``
        with the value as prefactor, where `j` refers to the boson or fermion
        species. If there are multiple species and `mu` is a `float`, it is
        applied as chemical potential to all particle species.
        `None` entries correspond to no chemical potential.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 1
    site = sites[0]
    h = 0
    if isinstance(mu, (list, np.ndarray)):
        msg = f'Invalid number of entries in `mu`: {len(mu)} != {site.num_species}'
        assert len(mu) == site.num_species, msg
        for i, mu_i in enumerate(mu):
            if mu_i is None:
                continue
            op_name = 'N' if site.num_species == 1 else f'N{i}'
            h -= mu_i * site.onsite_operators[op_name]
    else:
        op_name = 'N' if site.num_species == 1 else 'Ntot'
        h = -1 * mu * site.onsite_operators[op_name]
    if np.ndim(h) == 0:
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


def long_range_interaction(sites: list[BosonicDOF | FermionicDOF],
                           name: str = 'long-range interaction') -> Coupling:
    """Long-range density-density interactions for bosons or fermions.

    Corresponds to ``n_i n_j``, where `n_i` and `n_j` are the total onsite
    particle numbers (including all particle species) of the sites `i` and `j`.
    Sites `i` and `j` are assumed to be the first and the last entry in
    `sites`, respectively.
    """
    # TODO test that this builds what we expect
    assert len(sites) > 1, 'distance between the sites must be positive'
    sites_ij = [sites[0], sites[-1]]
    op_names = ['N' if site.num_species == 1 else 'Ntot' for site in sites_ij]
    ops = [site.onsite_operators[op_name] for site, op_name in zip(sites_ij, op_names)]
    h = ops[0]
    distance = len(sites) - 1
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
    assert len(sites) == 2
    return long_range_interaction(sites=sites, name=name)


def next_nearest_neighbor_interaction(sites: list[BosonicDOF | FermionicDOF],
                                      name: str = 'NNN interaction') -> Coupling:
    """Next nearest neighbor interactions for bosons or fermions.

    Corresponds to ``n_i n_j``, where `n_i` and `n_j` are the total onsite
    particle numbers (including all particle species) of the two next nearest
    neighboring sites.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 3
    return long_range_interaction(sites=sites, name=name)


def long_range_quadratic_coupling(sites: list[BosonicDOF | FermionicDOF],
                                  t: float | list[float | None] = 1.,
                                  delta: float | list[float | None] = None,
                                  name: str = 'long-range quad coupling') -> Coupling:
    r"""Long-range quadratic coupling (hopping and pairing) for bosons or fermions.

    Quadratic couplings generally take the form
    ``\sum_k -1 * t[k] a_i^\dagger(k) a_j(k) + delta[k] a_i^\dagger(k) a_j^\dagger(k) + h.c.``,
    where the first term corresponds to hopping processes and the second term to superconducting
    pairings. Here `a_i^\dagger(k)` and `a_j(k)` denote the creation and annihilation operators on
    sites `i` and `j`, respectively. Sites `i` and `j` are assumed to be the first and the last
    entry in `sites`, respectively. `k` denotes the boson / fermion species. Note the convention
    for the order of the operators (with ``j > i``). This is in particular important for fermions
    due to their anticommutation relations.

    Parameters
    ----------
    t : float | list[float | None]
        Hopping amplitudes. Add the corresponding hopping
        ``\sum_k -1 * t[k] a_i^\dagger(k) a_j(k) + h.c.`` with the value as prefactor, where `k`
        refers to the boson or fermion species. If there are multiple species and `t` is a `float`,
        it is applied as hopping amplitude to all particle species.
        `None` entries correspond to no hopping processes.
    delta : float | list[float | None]
        Superconducting pairing amplitudes. Add the corresponding pairing
        ``\sum_k delta[k] a_i^\dagger(k) a_j^\dagger(k) + h.c.`` with the value as prefactor, where
        `k` refers to the boson or fermion species. If there are multiple species and `delta` is a
        `float`, it is applied as pairing amplitude to all particle species.
        `None` entries correspond to no pairing processes.
    """
    # TODO test that this builds what we expect
    assert len(sites) > 1, 'distance between the sites must be positive'
    sites_ij = [sites[0], sites[-1]]
    assert sites_ij[0].num_species == sites_ij[1].num_species
    if not isinstance(t, (list, np.ndarray)):
        t = [t] * sites[0].num_species
    else:
        assert len(t) == sites[0].num_species
    if not isinstance(delta, (list, np.ndarray)):
        delta = [delta] * sites[0].num_species
    else:
        assert len(delta) == sites[0].num_species
    h = 0
    for k, (t_k, delta_k) in enumerate(zip(t, delta)):
        if t_k is None and delta_k is None:
            continue
        a_idk = sites_ij[0].creators[:, :, k]
        a_jk = sites_ij[1].annihilators[:, :, k]
        a_jdk = sites_ij[1].creators[:, :, k]
        for n, site in enumerate(sites):
            if isinstance(site, FermionicDOF):
                if n == 0:
                    jw_n = np.tensordot(site.creators[:, :, k:], site.annihilators[:, :, k:],
                                        axes=[[2, 1], [2, 0]])
                elif n == len(sites) - 1:
                    jw_n = np.tensordot(site.creators[:, :, :k], site.annihilators[:, :, :k],
                                        axes=[[2, 1], [2, 0]])
                else:
                    op_name = 'N' if site.num_species == 1 else f'Ntot'
                    jw_n = site.onsite_operators[op_name].to_numpy(understood_braiding=True)
                jw_n = np.diag(np.power(-1, np.diag(jw_n)))
            else:
                jw_n = np.diag(np.ones(a_idk.shape[0]))
            if n == 0:
                # JW acts on same site as a_i(k)
                h_k = a_idk @ jw_n
            elif n == len(sites) - 1:
                # multiply in the next step after the for loop
                pass
            else:
                h_k = np.tensordot(h_k, jw_n, axes=0)
        if t_k is not None:
            h -= t_k * np.tensordot(h_k, jw_n @ a_jk, axes=0)
        if delta_k is not None:
            h += delta_k * np.tensordot(h_k, jw_n @ a_jdk, axes=0)
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    # h has legs p0, p0*, p1, ..., but already has to correct signs for the correct leg order
    axes_perm = [2 * i for i in range(len(sites))]
    axes_perm.extend([i + 1 for i in reversed(axes_perm)])
    h = np.transpose(h, axes=axes_perm)
    # add hc for a_j^\dagger a_i
    h += np.conj(np.transpose(h, list(reversed(range(2 * len(sites))))))
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


def nearest_neighbor_hopping(sites: list[BosonicDOF | FermionicDOF],
                             t: float | list[float | None] = 1.,
                             name: str = 'NN hopping') -> Coupling:
    r"""Nearest neighbor hopping for bosons or fermions.

    Corresponds to ``-1 * t[k] a_i^\dagger(k) a_j(k) + h.c.``, where `a_i^\dagger(k)`
    and `a_j(k)` are the creation and annihilation operators on two nearest neighboring
    sites `i` and `j == i + 1`, respectively. `k` denotes the boson / fermion species.
    Note the convention for the order of the operators (with ``j > i``). This is in
    particular important for fermions due to their anticommutation relations.

    Parameters
    ----------
    t : float | list[float | None]
        Hopping amplitudes. Add the corresponding hopping ``-1 * t[k] a_i^\dagger(k) a_j(k) + h.c.``
        with the value as prefactor, where `k` refers to the boson or fermion species.
        If there are multiple species and `t` is a `float`, it is applied as hopping
        amplitude to all particle species.
        `None` entries correspond to no hopping processes.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 2
    return long_range_quadratic_coupling(sites=sites, t=t, delta=None, name=name)


def next_nearest_neighbor_hopping(sites: list[BosonicDOF | FermionicDOF],
                                  t: float | list[float | None] = 1.,
                                  name: str = 'NNN hopping') -> Coupling:
    r"""Next-nearest neighbor hopping for bosons or fermions.

    Corresponds to ``-1 * t[k] a_i^\dagger(k) a_j(k) + h.c.``, where `a_i^\dagger(k)`
    and `a_j(k)` are the creation and annihilation operators on two next-nearest
    neighboring sites `i` and `j == i + 2`, respectively. `k` denotes the boson / fermion
    species. Note the convention for the order of the operators (with ``j > i``). This is
    in particular important for fermions due to their anticommutation relations.

    Parameters
    ----------
    t : float | list[float | None]
        Hopping amplitudes. Add the corresponding hopping ``-1 * t[k] a_i^\dagger(k) a_j(k) + h.c.``
        with the value as prefactor, where `k` refers to the boson or fermion species.
        If there are multiple species and `t` is a `float`, it is applied as hopping
        amplitude to all particle species.
        `None` entries correspond to no hopping processes.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 3
    return long_range_quadratic_coupling(sites=sites, t=t, delta=None, name=name)


def onsite_sc_pairing(sites: list[BosonicDOF | FermionicDOF],
                      delta: float | list[list[float | None]] = 1.,
                      name: str = 'onsite SC pairing') -> Coupling:
    r"""Onsite superconducting pairing for bosons or fermions.

    Corresponds to ``\sum_{j,k} delta[j][k] a_i^\dagger(j) a_i^\dagger(k) + h.c.``,
    where `a_i^\dagger(j)` and `a_i^\dagger(k)` denote the onsite creation
    operators for boson / fermion species `j` and `k`, respectively. Note the
    convention for the order of the operators. This is in particular important
    for fermions due to their anticommutation relations.

    Parameters
    ----------
    delta : float | list[list[float | None]]
        Superconducting pairing amplitudes. Add the corresponding pairing
        ```\sum_{j,k} delta[j][k] a_i^\dagger(j) a_i^\dagger(k) + h.c.`` with
        the value as prefactor, where `j` and `k` refers to the boson or fermion
        species. Can in general be chosen to be a upper or lower triangular
        matrix (but this is not required). If there are multiple species and
        `delta` is a `float`, it is applied as pairing amplitude to all particle
        species.
        `None` entries correspond to no pairing processes.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 1
    if not isinstance(delta, (list, np.ndarray)):
        delta = delta * np.ones((sites[0].num_species, sites[0].num_species))
    else:
        delta = np.asarray(delta)
        assert delta.shape == (sites[0].num_species, sites[0].num_species)
    commute = -1 if isinstance(sites[0], FermionicDOF) else 1
    h = 0
    for k in range(sites[0].num_species):
        for j in range(k + 1):
            # combine contributions with the same operators
            if j == k:
                if delta[j, k] is None or isinstance(sites[0], FermionicDOF):
                    continue
                factor = delta[j, k]
            else:
                if delta[j, k] is None:
                    if delta[k, j] is None:
                        continue
                    factor = commute * delta[k, j]
                else:
                    factor = delta[j, k]
                    if delta[k, j] is not None:
                        factor += commute * delta[k, j]
            h_jk = np.copy(sites[0].creators[:, :, j])
            a_kd = sites[0].creators[:, :, k]
            if isinstance(sites[0], FermionicDOF):
                # onsite JW: product of species occupations from species 0 to species j - 1 / k - 1
                jw = np.tensordot(sites[0].creators[:, :, j:k], sites[0].annihilators[:, :, j:k],
                                  axes=[[2, 1], [2, 0]])
                jw = np.diag(np.power(-1, np.diag(jw)))
                h_jk = h_jk @ jw
            h += factor * h_jk @ a_kd
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    h += np.conj(np.transpose(h, [1, 0]))
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)
            

def nearest_neighbor_sc_pairing(sites: list[BosonicDOF | FermionicDOF],
                                delta: float | list[float | None] = 1.,
                                name: str = 'NN SC pairing') -> Coupling:
    r"""Nearest neighbor superconducting pairing for bosons or fermions.

    Corresponds to ``\sum_k delta[k] a_i^\dagger(k) a_j^\dagger(k) + h.c.``,
    where `a_i^\dagger(k)` and `a_j(k)^\dagger` denote the creation operators
    on two nearest neighboring sites `i` and `j == i + 1`, respectively. `k`
    denotes the boson / fermion species. Note the convention for the order of
    the operators (with ``j > i``). This is in particular important for fermions
    due to their anticommutation relations.

    Parameters
    ----------
    delta : float | list[float | None]
        Superconducting pairing amplitudes. Add the corresponding pairing
        ``\sum_k delta[k] a_i^\dagger(k) a_j^\dagger(k) + h.c.`` with the value
        as prefactor, where `k` refers to the boson or fermion species. If
        there are multiple species and `delta` is a `float`, it is applied as
        pairing amplitude to all particle species.
        `None` entries correspond to no pairing processes.
    """
    # TODO test that this builds what we expect
    assert len(sites) == 2
    return long_range_quadratic_coupling(sites=sites, t=None, delta=delta, name=name)


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
    backend = get_same_backend(*sites)
    device = sites[0].default_device
    assert all(s.default_device == device for s in sites[1:])
    labels = [f'p{i}' for i in range(len(sites))]
    labels = [*labels, *[f'{l}*' for l in reversed(labels)]]
    projector = SymmetricTensor.from_sector_projection(
        [s.leg for s in sites], sector=sector, backend=backend, labels=labels, device=device
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
