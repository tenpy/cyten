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
    SymmetricTensor, squeeze_legs, add_trivial_leg, permute_legs, compose, horizontal_factorization
)
from .degrees_of_freedom import (
    DegreeOfFreedom, SpinDOF, BosonicDOF, FermionicDOF, ClockDOF, ALL_SPECIES
)
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


def spin_spin_coupling(sites: list[SpinDOF], Jx: float = 0, Jy: float = 0, Jz: float = 0,
                       name: str = 'spin-spin') -> Coupling:
    r"""Two-site coupling between spins.

    .. math ::
        h_{ij} = \mathtt{Jx} S_i^x S_j^x + \mathtt{Jy} S_i^y S_j^y + \mathtt{Jz} S_i^z S_j^z

    Parameters
    ----------
    Jx, Jy, Jz: float
        Prefactor, as given above. By default, all prefactors vanish.
    """
    assert len(sites) == 2
    s1 = sites[0].spin_vector
    s2 = sites[1].spin_vector
    h = 0  # build in leg order [p0, p0*, p1, p1*] and transpose only once before returning
    h += Jx * np.tensordot(s1[:, :, 0], s2[:, :, 0], axes=0)
    h += Jy * np.tensordot(s1[:, :, 1], s2[:, :, 1], axes=0)
    h += Jz * np.tensordot(s1[:, :, 2], s2[:, :, 2], axes=0)
    h = np.transpose(h, [0, 2, 3, 1])
    return Coupling.from_dense_block(h, sites, name=name)


def spin_field_coupling(sites: list[SpinDOF], hx: float = 0, hy: float = 0, hz: float = 0,
                        name: str = 'spin-field') -> Coupling:
    r"""Single-site coupling of a spin to an external field.

    .. math ::
        h_i = \mathtt{hx} S_i^x + \mathtt{hy} S_i^y + \mathtt{hz} S_i^z

    Parameters
    ----------
    hx, hy, hz: float
        Prefactor, as given above. By default, all prefactors vanish.
    """
    assert len(sites) == 1
    s = sites[0].spin_vector
    h = hx * s[:, :, 0] + hy * s[:, :, 1] + hz * s[:, :, 2]
    return Coupling.from_dense_block(h, sites, name=name)


def aklt_coupling(sites: list[SpinDOF], J: float = 1, name: str = 'AKLT') -> Coupling:
    r"""Two-site AKLT coupling between spins.

    .. math ::
        h_{ij} = \mathtt{J} [\vec{S}_i \cdot \vec{S}_j + \frac{1}{3} (\vec{S}_i \cdot \vec{S}_j)^2]

    This is the coupling originally defined by Affleck, Kennedy, Lieb, Tasaki
    in :cite:`affleck1987`, except we drop the constant part of 1/3 per bond and rescale with a
    factor of 2, i.e. :math:`h_{ij} = 2 P^{S=2}_{i, j} + const.`.

    It was defined for spin-1 degrees of freedom in the original work, but we allow any site
    with a spin DOF. Note that the coupling simplifies to a Heisenberg coupling for spin-1/2.

    Parameters
    ----------
    J: float
        Prefactor, as given above. By default use ``1``.
    """
    assert len(sites) == 2
    s1 = sites[0].spin_vector
    s2 = sites[1].spin_vector
    S_dot_S = np.tensordot(s1, s2, axes=[2, 2])
    S_dot_S = np.transpose(S_dot_S, [0, 2, 3, 1])
    S_dot_S_square = np.tensordot(S_dot_S, S_dot_S, axes=[[3, 2], [0, 1]])
    h = J * (S_dot_S + S_dot_S_square / 3.)
    return Coupling.from_dense_block(h, sites, name=name)


def heisenberg_coupling(sites: list[SpinDOF], J: float = 1, name: str = 'S.S') -> Coupling:
    r"""Two-site Heisenberg coupling between spins.

    .. math ::
        h_{ij} = \mathtt{J} \vec{S}_i \cdot \vec{S}_j

    Parameters
    ----------
    J: float
        Prefactor, as given above. By default use ``1``, i.e. an anti-ferromagnetic coupling.
    """
    return spin_spin_coupling(sites=sites, Jx=J, Jy=J, Jz=J, name=name)


def chiral_3spin_coupling(sites: list[SpinDOF], chi: float = 1, name: str = 'S.SxS') -> Coupling:
    r"""Chiral coupling of three spins.

    .. math ::
        h_{ijk} = \mathtt{chi} \vec{S}_i \cdot ( \vec{S}_j \times \vec{S}_k )

    Parameters
    ----------
    chi: float
        Prefactor, as given above. By default use ``1``.
    """
    assert len(sites) == 3
    SxS = np.cross(sites[1].spin_vector[:, None, None, :, :],
                   sites[2].spin_vector[None, :, :, None, :],
                   axis=4)  # [p1, p2, p2*, p1*, i]
    h = chi * np.tensordot(sites[0].spin_vector, SxS, (-1, -1))  # [p0, p0*, p1, p2, p2*, p1*]
    h = np.transpose(h, [0, 2, 3, 4, 5, 1])
    return Coupling.from_dense_block(h, sites, name=name)


# BOSON AND FERMION COUPLINGS


def chemical_potential(sites: list[BosonicDOF] | list[FermionicDOF], mu: float,
                       species: int | list[int] = ALL_SPECIES, name: str = 'chem. pot.'
                       ) -> Coupling:
    r"""Chemical potential for bosons or fermions. Single-site coupling.

    .. math ::
        h_i = -\mathtt{mu} \sum_{k \in \mathtt{species} n_{i, k}

    where :math:`n_{i, k}` is the occupation number of species :math:`k` on site :math:`i`.

    Parameters
    ----------
    mu: float
        Chemical potential, as defined above.
    species: (list of) int, optional
        If given, the chemical potential only couples to the occupation of this species.
        By default, it couples to the total occupation of all species.
    """
    assert len(sites) == 1
    h = -mu * sites[0].get_occupation_numpy(species=species)
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def onsite_interaction(sites: list[BosonicDOF] | list[FermionicDOF], U: float = 1,
                       species: int = ALL_SPECIES, name: str = 'onsite interaction') -> Coupling:
    r"""Onsite interaction for bosons or fermions. Single-site coupling.

    .. math ::
        h_i = \frac{U}{2} n_i^2

    where :math:`n_i` is the total occupation number, or the occupation of a single `species`.

    Parameters
    ----------
    U: float
        Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
    species: int, optional
        If given, we use only the occupation of this one species as the density :math:`n_i`.
        By default, we use the total occupation of all species.
    """
    assert len(sites) == 1
    n_i = sites[0].get_occupation_numpy(species=species)
    h = .5 * U * n_i @ n_i
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def density_density_interaction(sites: list[BosonicDOF] | list[FermionicDOF], V: float = 1,
                                species_i: int = ALL_SPECIES, species_j: int = ALL_SPECIES,
                                name: str = 'density-density') -> Coupling:
    r"""Density-density interaction. Two-site coupling.

    .. math ::
        h_{ij} = \mathtt{V} n_i n_j

    where :math:`n_i` is the total occupation number.

    Parameters
    ----------
    V: float
        Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
    species_i, species_j: int, optional
        If given, we use only the occupation of this one species as the density :math:`n_{i/j}`.
        By default, we use the total occupation of all species.
        Note that if the two species are different, this coupling alone is not hermitian!
    """
    assert len(sites) == 2
    n_i = sites[0].get_occupation_numpy(species=species_i)
    n_j = sites[1].get_occupation_numpy(species=species_j)
    h = V * n_i[:, None, None, :] * n_j[None, :, :, None]  # [p0, p1, p1*, p0*]
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


def _quadratic_coupling_numpy(sites: list[BosonicDOF] | list[FermionicDOF], is_pairing: bool,
                              species) -> np.ndarray:
    """Create the numpy representation for both :func:`hopping` and :func:`pairing`."""
    site_i, site_j = sites
    species_i, species_j = species
    if species_i is ALL_SPECIES:
        species_i = [*range(site_i.num_species)]
    if species_j is ALL_SPECIES:
        species_j = [*range(site_j.num_species)]
    h = 0
    for k_i, k_j in zip(species_i, species_j):
        # since we work with numpy representations here, we need to consider JW strings.
        # visually (where columns represent different species)
        # |  site i   |  site j  |       |  site i   |  site j  |
        # | J J J O   |          |   =   |  op_i     |          |
        # | J J J J J | J J J O  |       |  JW_i     |  op_j    |
        op_i = site_i.get_creator_numpy(species=k_i, include_JW=True)

        # OPTIMIZE rm check?
        sign = -1 if isinstance(site_i, FermionicDOF) else +1
        assert np.allclose(op_i @ site_i._JW, sign * site_i._JW @ op_i)

        if is_pairing:
            op_j = site_i.get_creator_numpy(species=k_j, include_JW=True)
        else:
            op_j = site_i.get_annihilator_numpy(species=k_j, include_JW=True)
        h += (op_i @ site_i._JW)[:, None, None, :] * op_j[None, :, :, None]  # [p0, p1, p1*, p0*]
    return h + np.transpose(h.conj(), [3, 2, 1, 0])


def hopping(sites: list[BosonicDOF] | list[FermionicDOF], t: float = 1,
            species: tuple[list[int], list[int]] = (ALL_SPECIES, ALL_SPECIES),
            name: str = 'hopping') -> Coupling:
    r"""Hopping of fermions or bosons. Two-site coupling.

    .. math ::
        h_{ij} = -\mathtt{t} \sum_{k \in \mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j} + h.c.

    Parameters
    ----------
    t : float
        Prefactor, as given above. By default ``1``.
    species : tuple of list of int, optional
        Which species should participate (the sum above goes over ``k_i, k_j in zip(*species)``).
        By default, we let :math:`k_i = k_j` go over all species, i.e. include all
        "species preserving" hoppings.
    """
    h = -t * _quadratic_coupling_numpy(sites, is_pairing=False, species=species)
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def pairing(sites: list[BosonicDOF] | list[FermionicDOF], Delta: float = 1.,
            species: tuple[list[int], list[int]] = (ALL_SPECIES, ALL_SPECIES),
            name: str = 'pairing') -> Coupling:
    r"""Superconducting pairing of fermions or bosons. Two-site coupling.

    .. math ::
        h_{ij} = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j}^\dagger + h.c.

    .. note ::
        This coupling assumes distinct sites :math:`i \neq j`.
        Use :func:`onsite_pairing` for :math:`i = j`.

    Parameters
    ----------
    Delta : float
        Prefactor, as given above. By default ``1``.
    species : tuple of list of int, optional
        Which species should participate (the sum above goes over ``k_i, k_j in zip(*species)``).
        By default, we let :math:`k_i = k_j` go over all species, i.e. include all "same-species"
        pairings.

    See Also
    --------
    onsite_pairing
    """
    h = Delta * _quadratic_coupling_numpy(sites, is_pairing=True, species=species)
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def onsite_pairing(sites: list[BosonicDOF] | list[FermionicDOF], Delta: float = 1.,
                   species: tuple[list[int], list[int]] = (ALL_SPECIES, ALL_SPECIES),
                   name: str = 'onsite pairing') -> Coupling:
    r"""Superconducting pairing of fermions or bosons. Single-site coupling.

    .. math ::
        h_i = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_1}^\dagger a_{i, k_2}^\dagger + h.c.

    Parameters
    ----------
    Delta : float
        Prefactor, as given above. By default ``1``.
    species : tuple of list of int, optional
        Which species should participate (the sum above goes over ``k_1, k_2 in zip(*species)``).
        By default, we let :math:`k_1 = k_2` go over all species, i.e. include all "same-species"
        pairings.

    See Also
    --------
    pairing
    """
    site, = sites
    species_1, species_2 = species
    if species_1 is ALL_SPECIES:
        species_1 = [*range(site.num_species)]
    if species_2 is ALL_SPECIES:
        species_2 = [*range(site.num_species)]
    h = 0
    for k_1, k_2 in zip(species_1, species_2, strict=True):
        a_i_hc = site.get_creator_numpy(species=k_1, include_JW=True)
        a_j_hc = site.get_creator_numpy(species=k_2, include_JW=True)
        h += Delta * a_i_hc @ a_j_hc
    h + np.transpose(h.conj(), [3, 2, 1, 0])
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


# CLOCK COUPLINGS

def clock_clock_coupling(sites: list[ClockDOF], Jx: float = 0, Jz: float = 0,
                         name: str = 'clock-clock') -> Coupling:
    r"""Two-site coupling between quantum clocks.

    .. math ::
        h_{ij} = \mathtt{Jx} X_i X_j^\dagger + \mathtt{Jz} Z_i Z_j^\dagger + h.c.

    Parameters
    ----------
    Jx, Jz: float
        Prefactor, as given above. By default, all prefactors vanish.
    """
    assert len(sites) == 2
    X_i = sites[0].clock_operators[:, :, 0]
    Z_i = sites[0].clock_operators[:, :, 1]
    X_j = sites[1].clock_operators[:, :, 0]
    Z_j = sites[1].clock_operators[:, :, 1]
    h = Jx * X_i[:, None, None, :] * X_j.T.conj()[None, :, :, None]  # [p0, p1, p1*, p0*]
    h += Jz * Z_i[:, None, None, :] * Z_j.T.conj()[None, :, :, None]
    h = h + np.transpose(h.conj(), [3, 2, 1, 0])
    return Coupling.from_dense_block(h, sites, name=name)


def clock_field_coupling(sites: list[ClockDOF], hx: float = None, hz: float = None,
                         name: str = 'clock-field') -> Coupling:
    r"""Single-site coupling of a quantum clock to an external field.

    .. math ::
        h_i = \mathtt{hx} X_i + \mathtt{hz} Z_i + h.c.

    Parameters
    ----------
    hx, hz: float
        Prefactor, as given above. By default, all prefactors vanish.
    """
    assert len(sites) == 1
    X = sites[0].clock_operators[:, :, 0]
    Z = sites[0].clock_operators[:, :, 1]
    h = hx * (X + X.T.conj()) + hz * (Z + Z.T.conj())
    return Coupling.from_dense_block(h, sites, name=name)


# ANYONIC COUPLINGS


def sector_projection_coupling(sites: list[DegreeOfFreedom], J: float, sector: Sector, name: str
                               ) -> Coupling:
    """Coupling that is given by the projector onto a single sector.

    The number of sites is arbitrary and the operator :math:`h_{ij...}` is given
    by :meth:`cyten.SymmetricTensor.from_sector_projection`, with prefactor `J`.
    Note that positive `J` mean that states that fuse to the given `sector` are energetically
    *disfavored*.
    """
    backend = get_same_backend(*sites)
    device = sites[0].default_device
    assert all(s.default_device == device for s in sites[1:])
    labels = [f'p{i}' for i in range(len(sites))]
    labels = [*labels, *[f'{l}*' for l in reversed(labels)]]
    projector = SymmetricTensor.from_sector_projection(
        [s.leg for s in sites], sector=sector, backend=backend, labels=labels, device=device
    )
    return Coupling.from_tensor(projector, sites=sites, name=name)


def gold_coupling(sites: list[GoldenSite], J: float = 0, name: str = 'P_vac') -> Coupling:
    r"""Two-site coupling of Fibonacci anyons that energy splits fusion to vacuum or tau.

    .. math ::
        h_{ij} = -J P^\text{vac}_{i, j}

    Parameters
    ----------
    J: float
        Prefactor, as given above. By default ``1``. Positive `J` energetically favor the
        trivial fusion channel, i.e. they are the "antiferromagnetic" analog.
    """
    assert len(sites) == 2
    for site in sites:
        assert isinstance(site.symmetry, FibonacciAnyonCategory)
        assert site.leg.sector_decomposition_where(FibonacciAnyonCategory.tau) is not None
    return sector_projection_coupling(sites, sector=FibonacciAnyonCategory.vacuum, name=name)
