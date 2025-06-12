import numpy as np
from typing import Literal

from ..spaces import ElementarySpace
from ..tensors import SymmetricTensor
from ..symmetries import SU2Symmetry, U1Symmetry, ZNSymmetry, NoSymmetry, FibonacciAnyonCategory


# ------------------------------------------------------------------------------------------
# Degree of freedom classes : know about local operator algebra
# ------------------------------------------------------------------------------------------


class DOF:
    # TODO name choice?? Maybe SpinAlgebra(OperatorAlgebra)?
    leg: ElementarySpace
    local_ops: dict[str, SymmetricTensor]  # e.g. MPS.expectation_value can look-up names here


class SpinDOF(DOF):
    def __init__(self, leg: ElementarySpace, spin_vector: np.ndarray, which_symmetry: int = 0):
        assert spin_vector.shape == (3, leg.dim, leg.dim)
        self.spin_vector = spin_vector
        self.spin_symmetry = spin_symmetry = leg.symmetry.factors[which_symmetry]  # dummy: what if symmetry is not a product
        # more operators?
        match spin_symmetry:
            case SU2Symmetry():
                spin_sector = leg.sector_decomposition[0, which_symmetry]
                # dummy consider ProdSymmetry.sector_slices, or what if not a ProdSymmetry
                assert np.all(leg.sector_decomposition[:, which_symmetry] == spin_sector)
                local_ops = dict()
            case U1Symmetry():
                # todo: sanity check that all charge values appear the same # of times
                local_ops = dict(Sz=spin_vector[-1])
            case ZNSymmetry(N=2), NoSymmetry():
                local_ops = dict(Sx=spin_vector[0], Sy=spin_vector[1], Sz=spin_vector[2])
                # TODO: should we let the DOF always know what 'Sx', ... is?
                #       that way "forbidden by symmetry" is different error than "misspelled".
            case _:
                raise ValueError
        DOF.__init__(self, leg, local_ops)


class FermionDOF(DOF):
    def __init__(self, leg: ElementarySpace, creator: np.ndarray, annihilator: np.ndarray):
        self.leg = leg
        self.creator = creator
        self.annihilator = annihilator
        N = creator @ annihilator
        # ... define more operators?
        self.number_symmetry = None  # dummy, similar to SpinDOF
        local_ops = dict(N=N)  # dummy, put more operators
        DOF.__init__(self, leg, local_ops)


class GoldenDOF(DOF):
    def __init__(self, leg: ElementarySpace, which_symmetry: int = 0):
        self.fib_symmetry = fib_symmetry = leg.symmetry.factors[which_symmetry]  # dummy: what if symmetry is not a product
        assert isinstance(fib_symmetry, FibonacciAnyonCategory)

        # dummy: this check needs to be more involved if there are additional symmetries
        assert leg.sector_multiplicity(fib_symmetry.vacuum) == leg.sector_multiplicity(fib_symmetry.tau)

        DOF.__init__(self, leg)


# ------------------------------------------------------------------------------------------
# Sites: similar role to tenpy v1, but we have the DOF classes instead of local operators.
# ------------------------------------------------------------------------------------------


class Site:
    leg: ElementarySpace
    degrees_of_freedom: dict[str, DOF]


class SpinSite(Site):
    spin: float = .5
    conserve: Literal['SU(2)', 'Sz', 'parity', 'None']

    def __init__(self, spin, *args):
        d = 2 * spin + 1
        sx = sy = sz = np.zeros((d, d), complex)  # dummy
        spin_vector = np.array([sx, sy, sz])
        leg = ElementarySpace(...)  # dummy: sectors depend on conserve
        degrees_of_freedom = dict(
            spin=SpinDOF(leg, spin_vector),
        )
        Site.__init__(self, leg=leg, degrees_of_freedom=degrees_of_freedom)


class SpinHalfFermionSite(Site):
    conserve_S: Literal['SU(2)', 'Sz', 'parity', 'None'] = None
    conserve_N: Literal['N', 'parity', 'None'] = None

    def __init__(self, *args):
        leg = ElementarySpace(...)  # dummy: sectors depend on conserve
        spin_vector = creator = annihilator = np.zeros((4, 4), complex)  # dummy, do actual matrices
        degrees_of_freedom = dict(
            spin=SpinDOF(leg, spin_vector),
            fermion_occupation=FermionDOF(leg, creator, annihilator)
        )
        Site.__init__(self, leg=leg, degrees_of_freedom=degrees_of_freedom)


class GoldenSite(Site):
    def __init__(self, handedness):
        symmetry = FibonacciAnyonCategory(handedness)
        leg = ElementarySpace.from_basis(symmetry, [symmetry.vacuum, symmetry.tau])
        degrees_of_freedom = dict(
            golden=GoldenDOF(leg)
        )
        Site.__init__(self, leg=leg, degrees_of_freedom=degrees_of_freedom)


# ------------------------------------------------------------------------------------------
# Couplings : multi-site operators that are used to build Hamiltonian MPOs,
# can measure correlation functions etc.
# ------------------------------------------------------------------------------------------


class Coupling:
    sites: list[Site]
    factorization: list[SymmetricTensor]  # [vL, p, vR, p*]

    def _init_from_numpy(self, operator: np.ndarray, sites: list[Site]):
        ...

    def _init_from_tensor(self, operator: SymmetricTensor, sites: list[Site]):
        ...


class OnSiteOperator(Coupling):
    operator: SymmetricTensor


class HeisenbergCoupling(Coupling):
    """S_i dot S_j"""
    def __init__(self, site1: Site, site2: Site, dof1='spin', dof2='spin'):
        assert site1.leg.symmetry == site2.leg.symmetry
        s1: SpinDOF = site1.degrees_of_freedom[dof1]
        s2: SpinDOF = site2.degrees_of_freedom[dof2]
        h = np.tensordot(s1.spin_vector, s2.spin_vector, (0, 0))  # [p0, p0*, p1, p1*]
        h = np.transpose(h, [0, 2, 1, 3])  # [p0, p1, p0*, p1*]
        Coupling._init_from_numpy(h, [site1, site2])


class ChiralSpinCoupling(Coupling):
    """S_i dot (S_j cross S_k)"""
    def __init__(self, site1: Site, site2: Site, site3: Site,
                 dof1='spin', dof2='spin', dof3='spin'):
        assert site1.leg.symmetry == site2.leg.symmetry == site3.leg.symmetry
        s1: SpinDOF = site1.degrees_of_freedom[dof1]
        s2: SpinDOF = site2.degrees_of_freedom[dof2]
        s3: SpinDOF = site3.degrees_of_freedom[dof3]

        # dummy: should be cross product
        S2_cross_S3 = s2.spin_vector[:, :, None, :, None] * s3.spin_vector[:, None, :, None, :]

        h = np.tensordot(s1.spin_vector, S2_cross_S3, (0, 0))
        h = np.transpose(h, [0, 2, 1, 3, 4, 5])  # dummy: leg order is wrong
        Coupling._init_from_numpy(h, [site1, site2, site3])


class GoldenCoupling(Coupling):
    """projects two sites onto the tau fusion channel"""
    def __init__(self, site1: Site, site2: Site, dof1='golden', dof2='golden'):
        assert site1.leg.symmetry == site2.leg.symmetry
        tau_sector = [1]

        # this classmethod doesnt exist yet, would be light wrapper around from_sector_block_func
        h = SymmetricTensor.from_sector_projection([site1.leg, site2.leg], tau_sector)

        # in this case, we dont actually need the DOF, correct? the symmetry is already enough.
        Coupling._init_from_tensor(h)


# ------------------------------------------------------------------------------------------
# Helpers for name lookups
# ------------------------------------------------------------------------------------------


coupling_aliases = {
    HeisenbergCoupling: ['S.S', 'Heisenberg'],
    ChiralSpinCoupling: ['S.SxS'],
    ...: ['S+S-'],
    ...: ['SzSz'],
    ...: ['fermion_hop', 'CdC'],
    ...: ['boson_hop', 'BdB'],
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
    if isinstance(cls, str):
        cls = coupling_aliases_lookup.get(cls, None)
        if cls is None:
            cls = find_subclass(Coupling, cls)
    assert issubclass(cls, Coupling)
    return cls(*sites)
