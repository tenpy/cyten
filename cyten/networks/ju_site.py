from __future__ import annotations
import numpy as np
from typing import Literal, ClassVar

from ..spaces import ElementarySpace
from ..tensors import SymmetricTensor
from ..symmetries import SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory, ProductSymmetry


# ==========================================================================================
# Design constraints / criteria
# ==========================================================================================
# factorized form:
#       eventually we need to extract an MPO-like form for multi-site operators
#       e.g. when building hamiltonian/WI/WII MPOs, or for correlation_function etc
# short identifiers:
#       We want a system, similar to tenpy v1, where we can refer to operators by a short and
#       intuitively readable (string?) identifier.
#       TBD: what is the scope/namespace where they must be unique?
#            per site-type? per model? globally in cyten?
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






# ==========================================================================================
# Sites: similar role to tenpy v1, but we only have symmetric onsite op
# ==========================================================================================

# Base class
# ------------------------------------------------------------------------------------------


class Site:
    state_labels: list[str]
    leg: ElementarySpace
    onsite_operators: dict[str, OnSiteOperator]


# Intermediate "interface" classes that promise a certain operator algebra::
# ------------------------------------------------------------------------------------------


class SpinfulSite(Site):
    spin_vector: np.ndarray  # [p, p*, i]  ,  ==  [Sx, Sy, Sz]
    spin_conserve: Literal['SU(2)', 'Sz', 'parity', 'None']
    spin_symmetry: SU2Symmetry | U1Symmetry | ZNSymmetry | NoSymmetry
    # adds e.g. onsite_operators['Sz'] if symmetry allows it and so on


class FermionicSite(Site):
    # TODO similarly for bosons...
    creators: np.ndarray  # [p, p*, i] where i are different species ;  == [Cd0, Cd1, ...]
    annihilators: np.ndarray  # [p, p*, i] ;  == [C0, C1, ...]
    fermion_species_labels: list[str]  # [i]
    fermion_conserve: Literal['N', 'parity', 'None']
    fermion_symmetry: FermionOccupation | FermionParity | NoSymmetry
    # defines e.g. ``N = sum_k Cd_k C``


# Concrete classes::
# ------------------------------------------------------------------------------------------


class SpinSite(SpinfulSite):
    spin: float = .5



# TODO maybe do composition instead of diamond inheritance?
class SpinHalfFermionSite(SpinfulSite, FermionicSite):
    ...


class SpinLessFermionSite(FermionicSite):
    ...


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
    aliases: tuple[str, ...] = ()  # if couplings are subclasses: classvar, otherwise regular attr
    sites: list[Site]
    factorization: list[SymmetricTensor]  # [vL, p, vR, p*]

    def _init_from_numpy(self, operator: np.ndarray, sites: list[Site]):
        ...

    def _init_from_tensor(self, operator: SymmetricTensor, sites: list[Site]):
        ...


class OnSiteOperator(Coupling):
    operator: SymmetricTensor  # [p, p*]


class HeisenbergCoupling(Coupling):
    """S_i dot S_j"""
    def __init__(self, site1: SpinfulSite, site2: SpinfulSite):
        assert isinstance(site1, SpinfulSite)
        assert isinstance(site2, SpinfulSite)
        assert site1.leg.symmetry == site2.leg.symmetry
        h = np.tensordot(site1.spin_vector, site2.spin_vector, (0, 0))  # [p0, p0*, p1, p1*]
        h = np.transpose(h, [0, 2, 1, 3])  # [p0, p1, p0*, p1*]
        Coupling._init_from_numpy(operator=h, sites=[site1, site2])


class ChiralSpinCoupling(Coupling):
    """S_i dot (S_j cross S_k)"""
    def __init__(self, site1: SpinfulSite, site2: SpinfulSite, site3: SpinfulSite):
        assert isinstance(site1, SpinfulSite)
        assert isinstance(site2, SpinfulSite)
        assert isinstance(site3, SpinfulSite)
        assert site1.leg.symmetry == site2.leg.symmetry == site3.leg.symmetry
        S2_cross_S3 = np.cross(site1.spin_vector, site2.spin_vector, axis=-1)
        h = np.tensordot(site1.spin_vector, S2_cross_S3, (0, 0))
        h = np.transpose(h, [0, 2, 1, 3, 4, 5])  # dummy: leg order is wrong
        Coupling._init_from_numpy(operator=h, sites=[site1, site2, site3])


class GoldenCoupling(Coupling):
    """projects two sites onto the tau fusion channel"""
    def __init__(self, site1: GoldenSite, site2: GoldenSite):
        assert isinstance(site1, GoldenSite)
        assert isinstance(site2, GoldenSite)
        assert site1.leg.symmetry == site2.leg.symmetry
        tau_sector = [1]

        # this classmethod doesnt exist yet, would be light wrapper around from_sector_block_func
        h = SymmetricTensor.from_sector_projection([site1.leg, site2.leg], tau_sector)

        Coupling._init_from_tensor(h)


class FermionHopping(Coupling):
    """sum_k c_{i,k}^\dagger c_{j,k}

    i.e. hopping of all fermion species with same amplitude.

    TODO generalize to hopping of individual species here, by giving prefactors, or in new
    class(es)?
    """
    def __init__(self, site1: FermionicSite, site2: FermionicSite, plus_hc: bool = True):
        assert isinstance(site1, FermionicSite)
        assert isinstance(site2, FermionicSite)
        h = np.tensordot(site1.creators, site2.annihilators, (-1, -1))  # [p1, p1*, p2, p2*]
        h = np.transpose(h, [0, 2, 1, 3])
        if plus_hc:
            h = h + h.conj().transpose([2, 3, 0, 1])
        Coupling._init_from_tensor(h)


# ==========================================================================================
# Helpers for name lookups
# ==========================================================================================


coupling_aliases = {
    HeisenbergCoupling: ['S.S', 'Heisenberg'],
    ChiralSpinCoupling: ['S.SxS'],
    ...: ['S+S-'],
    ...: ['SzSz'],
    FermionHopping: ['fermion_hop', 'CdC'],  # todo establish rules for a mini-language?
    ...: ['boson_hop', 'BdB'],
    ...: ...,
}


coupling_aliases_lookup = {
    alias: cls for cls, aliases in coupling_aliases.items() for alias in aliases
}


def add_coupling_alias(cls: type[Coupling], alias: str, *more: str):
    assert issubclass(cls, Coupling)
    assert alias not in coupling_aliases_lookup
    assert all(a not in coupling_aliases_lookup for a in more)
    aliases = coupling_aliases.get(cls, [])
    aliases.append(alias)
    aliases.extend(more)
    coupling_aliases[cls] = aliases
    coupling_aliases_lookup[alias] = cls
    for a in more:
        coupling_aliases_lookup[a] = cls


# used e.g. when building MPOs in Hamiltonians, or in MPS.correlation_function
def get_coupling(cls: str | type[Coupling], sites: list[Site]):
    # TODO think about caching here
    if isinstance(cls, str):
        cls = coupling_aliases_lookup.get(cls, None)
        if cls is None:
            cls = find_subclass(Coupling, cls)
    assert issubclass(cls, Coupling)
    return cls(*sites)


# FIXME make model mockups!
