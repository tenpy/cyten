from __future__ import annotations
import numpy as np
from typing import Literal

from ..spaces import ElementarySpace
from ..tensors import SymmetricTensor
from ..symmetries import SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory, ProductSymmetry


# ==========================================================================================
# Sites: similar role to tenpy v1, but we only have symmetric onsite op
# ==========================================================================================

# Base class
# ------------------------------------------------------------------------------------------


class Site:
    state_labels: list[str]
    leg: ElementarySpace
    onsite_operators: dict[str, OnSiteOperator]
    
    def __init__(self, leg: ElementarySpace,
                 onsite_operators: dict[str, np.ndarray | SymmetricTensor | OnSiteOperator]
                 ):
        ...


# Intermediate "interface" classes that promise a certain operator algebra::
# ------------------------------------------------------------------------------------------


class SpinfulSite(Site):
    spin_vector: np.ndarray  # [p, p*, i]
    spin_conserve: Literal['SU(2)', 'Sz', 'parity', 'None']
    spin_symmetry: SU2Symmetry | U1Symmetry | ZNSymmetry | NoSymmetry
    # adds e.g. onsite_operators['Sz'] if symmetry allows it and so on

    def __init__(self,
                 leg: ElementarySpace,
                 which_symmetry: int | None,  # such that leg.symmetry[which_symmetry] is the spin symm
                 spin_conserve: Literal['SU(2)', 'Sz', 'parity', 'None'],
                 spin_vector: np.ndarray,
                 onsite_operators: dict[str, np.ndarray | SymmetricTensor | OnSiteOperator]):
        assert spin_vector.shape == (leg.dim, leg.dim, 3)
        if spin_conserve != 'SU(2)':
            self.Sz = spin_vector[:, :, 2]
        ...


class FermionicSite(Site):
    # TODO similarly for bosons...
    creators: np.ndarray  # [p, p*, i] where i are different species
    annihilators: np.ndarray  # [p, p*, i]
    fermion_species_labels: list[str]  # [i]
    fermion_conserve: Literal['N', 'parity', 'None']
    fermion_symmetry: FermionOccupation | FermionParity | NoSymmetry

    def __init__(self,
                 leg: ElementarySpace,
                 which_symmetry: int | None,  # such that leg.symmetry[which_symmetry] is the spin symm
                 fermion_conserve: Literal['N', 'parity', 'None'],
                 creators: np.ndarray, annihilators: np.ndarray,
                 fermion_species_labels: list[str],
                 onsite_operators: dict[str, np.ndarray | SymmetricTensor | OnSiteOperator]
                 ):
        self.num_species = Nk = creators.shape[-1]
        assert creators.shape == (leg.dim, leg.dim, Nk)
        assert annihilators.shape == (leg.dim, leg.dim, Nk)
        # N = \sum_k Cd_k C_k    ;   N[p, p*]  = sum_k sum_q Cd[p, q, k] C[q, p*, k]
        self.fermion_N = np.tensordot(creators, annihilators, ([1, 2], [0, 2]))
        if fermion_conserve == 'None':
            raise NotImplementedError  # TODO need to deal with JW strings!
        ...


# Concrete classes::
# ------------------------------------------------------------------------------------------


class SpinSite(SpinfulSite):
    def __init__(self, spin: float = .5, spin_conserve: Literal['SU(2)', 'Sz', 'parity', 'None'] = 'None'):
        ...



# TODO maybe do composition instead of diamond inheritance?
class SpinHalfFermionSite(SpinfulSite, FermionicSite):
    def __init__(self,
                 spin_conserve: Literal['SU(2)', 'Sz', 'parity', 'None'],
                 fermion_conserve: Literal['N', 'parity', 'None'],
                 ):
        ...


class SpinLessFermionSite(FermionicSite):
    def __init__(self,
                 fermion_conserve: Literal['N', 'parity', 'None'],
                 ):
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
