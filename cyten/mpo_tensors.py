"""Tensors with the block-triangular structure that arises in finite state machine MPOs"""
# Copyright (C) TeNPy Developers, Apache license

from typing import Sequence

from .spaces import ElementarySpace, TensorProduct
from .tensors import SymmetricTensor


class MPOTensor:
    r"""A block upper-triangular tensor that arises in Hamiltonian MPOs

    The tensor has four legs ``[vL, p, vR, p*]``.
    Each sector of the virtual legs ``vL`` and *independently* ``vR`` corresponds to a state of
    the finite state machine, that is we have

    .. math ::
        V_L = I_\mathtt{IdL} \oplus S_L \oplus I_\mathtt{IdR}
        V_R = I_\mathtt{IdL} \oplus S_R \oplus I_\mathtt{IdR}

    In particular, we enforce that there are (at least) two copies of the trivial sector with
    canonical labels ``IdL, IdR``, and further sectors.

    An MPOTensor is then constrained to the following block upper-triangular form as a matrix
    of the ``vL, vR`` legs::

        [[ 1  A  B ]
         [ 0  C  D ]
         [ 0  0  1 ]]

    and we store these entries explicitly as :attr:`A`, :attr:`B`, :attr:`C`, :attr:`D`.

    Attributes
    ----------
    A : SymmetricTensor
        A block of the tensors. Legs ``[IdL, p] <- [p, SR]``.
    B : SymmetricTensor
        A block of the tensors. Legs ``[IdL, p] <- [p, IdR]``.
    C : SymmetricTensor
        A block of the tensors. Legs ``[SL, p] <- [p, SR]``.
    D : SymmetricTensor
        A block of the tensors. Legs ``[SL, p] <- [p, IdR]``.
    vL_states, vR_states : list of str
        Labels ``['IdL', ..., 'IdR']`` for the sectors of the virtual legs.
        OPTIMIZE should we worry about this getting too large...? worst case 1000 strings?

    Notes
    -----
    TODO
        - subclass Tensor or not?
        - name of the class?
        - similar class for WI / WII MPOs ? generalize or just have two similar structures?
        - variable names for the 4 blocks?
        - variable names? e.g. of the spaces that C has as legs
        - IdL / IdR correct or accidentally swapped?
    """

    def __init__(self,
                 A: SymmetricTensor, B: SymmetricTensor, C: SymmetricTensor, D: SymmetricTensor,
                 vL_states: Sequence[str] = None, vR_states: Sequence[str] = None):
        vL, vR, p = self.verify_tensors(A, B, C, D)
        self.codomain = TensorProduct([vL, p])
        self.domain = TensorProduct([p, vR])
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    @staticmethod
    def verify_tensors(A: SymmetricTensor, B: SymmetricTensor, C: SymmetricTensor, D: SymmetricTensor):
        """Check if tensors are consistent."""
        symmetry = A.symmetry
        for T in [A, B, C, D]:
            assert T.codomain_labels == ['vL', 'p']
            assert T.domain_labels == ['p*', 'vR']
            assert T.symmetry == symmetry

        p_leg = A.get_leg('p')
        SL_leg = C.get_leg('vL')
        SR_leg = C.get_leg_co_domain('vR')
        if any(not isinstance(leg, ElementarySpace) for leg in [p_leg, SL_leg, SR_leg]):
            # should we allow this? does it cause any problems later??
            raise NotImplementedError

        for T in [A, B, C, D]:
            assert T.get_leg('p') == p_leg
            assert T.get_leg_co_domain('p*') == p_leg

        assert A.get_leg('vL').is_trivial
        assert A.get_leg_co_domain('vR') == SR_leg

        assert B.get_leg('vL').is_trivial
        assert B.get_leg_co_domain('vR').is_trivial

        assert C.get_leg('vL') == SL_leg
        assert C.get_leg_co_domain('vR') == SR_leg

        assert D.get_leg('vL') == SL_leg
        assert D.get_leg_co_domain('vR').is_trivial

        Id = ElementarySpace.from_trivial_sector(symmetry=symmetry)
        vL = ElementarySpace.direct_sum(Id, SL_leg, Id)
        vR = ElementarySpace.direct_sum(Id, SR_leg, Id)
        return vL, vR, p_leg
