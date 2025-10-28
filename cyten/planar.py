"""TODO"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Sequence

from .tensors import Tensor, tdot, _dual_leg_label
from .sparse import LinearOperator
from .tools import duplicate_entries


class LegPlaceholder:
    """Helper object to define where a single leg of a :class:`TensorPlaceholder` connects.

    We have ``T[label] == LegPlaceholder(T, label)`` and can then do e.g.
    ``T['a'] @ X['b']`` for a contraction of ``T['c'] @ 'c0'`` for an open leg.

    TODO elaborate
    """

    def __init__(self, parent: TensorPlaceholder, label: str):
        self.parent = parent
        self.label = label

    def contract(self, other: LegPlaceholder):
        self.parent.add_contracted_leg(self, other)
        other.parent.add_contracted_leg(other, self)

    def open(self, label: str):
        self.parent.add_open_leg(self, label)

    def __matmul__(self, other):
        if isinstance(other, LegPlaceholder):
            self.contract(other)
        elif isinstance(other, str):
            self.open(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, str):
            self.open(other)
        else:
            return NotImplemented


class TensorPlaceholder:
    """Placeholder for a tensor used to define :class:`PlanarDiagram` s.

    Attributes
    ----------
    labels
        The labels of the tensor (up to cyclic permutation). This means that as long as we go
        clockwise around the shape, any starting point can be chosen for the labels.
    dims
        For each leg (given in the same order as the :attr:`labels`), a symbol that represents the
        dimension of that leg, e.g. commonly ``'chi'`` for virtual MPS legs.
        Contraction costs can be expressed as polynomials in these symbols.
    contracted_legs : {str : LegPlaceholder}
        Specifies the contractions that are already 
    open_legs
    """

    def __init__(self, labels: Sequence[str], dims: Sequence[str] = None):
        assert not duplicate_entries(labels), 'labels must be unique'
        self.labels = labels  # expected labels in the tensor
        self.dims = dims  # symbols to express scaling of costs, e.g. 'chi', 'd'
        self.contracted_legs: dict[str, LegPlaceholder] = {}
        self.open_legs: dict[str, str] = {}

    def add_contracted_leg(self, leg: LegPlaceholder, other_leg: LegPlaceholder):
        self.contracted_legs[leg.label] = other_leg

    def add_open_leg(self, leg: LegPlaceholder, label: str):
        self.open_legs[leg.label] = label

    def check_compatible_with(self, tensor: Tensor) -> bool:
        """Check if the tensor matches the placeholder, raise otherwise."""
        if not tensor.is_fully_labelled:
            raise ValueError('Tensor must be fully labelled')
        if tensor.num_legs != len(self.labels):
            raise ValueError('Mismatching number of legs')
        if not tensor.labels_are(*self.labels):
            raise ValueError('Mismatching labels')
        offset = tensor.get_leg_idcs(self.labels[0])
        if tensor._labels != self.labels[-offset:] + self.labels[:-offset]:
            raise ValueError(f'Labels must match placeholder up to cyclical permutation. '
                             f'Expected {", ".join(self.labels)}. Got {", ".join(tensor._labels)}')

    def copy(self):
        return TensorPlaceholder(labels=self.labels[:], dims=self.dims[:])

    def dagger(self):
        return TensorPlaceholder(labels=[_dual_leg_label(l) for l in reversed(self.labels)],
                                 dims=self.dims[::-1])

    @property
    def fully_specified(self) -> bool:
        """If the connectivity of all legs is specified.

        Fully specified means that each leg, or rather its :attr:`label` appears in either
        the :attr:`contracted_legs` or the :attr:`open_legs`.
        """
        for l in self.labels:
            if l not in self.contracted_legs and l not in self.open_legs:
                return False
        return True

    def __getitem__(self, idx):
        if not isinstance(idx, str):
            msg = f'Expected string label. Got {type(idx)}'
            raise TypeError(msg)
        if idx not in self.labels:
            msg = f'Invalid label: "{idx}". Expected one of {", ".join(self.labels)}'
            raise ValueError(msg)
        return LegPlaceholder(self, idx)


class PlanarDiagram(metaclass=ABCMeta):
    """TODO: mention "access points". 

    Pitfalls / warnings::
        - clockwise order of labels

    """

    @classmethod
    @abstractmethod
    def define_diagram(cls) -> list[TensorPlaceholder]:
        """TODO"""
        ...

    @classmethod
    def evaluate(cls, *tensors: Tensor) -> Tensor:
        """Perform the contraction specified by the diagram.

        Parameters
        ----------
        tensors : list of Tensor
            The tensors in the diagram. Must
        """
        # Note: subclasses should probably override this with a thin wrapper that has a named
        #       argument for each tensor, to make the order explicit.
        tensors = list(tensors)
        placeholders = cls.define_diagram()
        order = cls.get_contraction_order(tensors=tensors, placeholders=placeholders)
        return contract(tensors=tensors, placeholders=placeholders, order=order)

    @classmethod
    def get_contraction_order(cls, tensors: list[Tensor], placeholders: list[TensorPlaceholder]
                              ) -> list[int]:
        """Determine the order for binary contractions during :meth:`do_binary_contractions`.

        Returns
        -------
        order : list of int
            Indicates the order in which tensors are contracted, i.e. the first binary contraction
            is between ``tensors[order[0]]`` and ``tensors[order[1]]``.
            We then contract that result with ``tensors[order[2]]`` and so on.
        """
        # FIXME dummy. The default here should call some greedy optimizer.
        return [*range(len(tensors))]




class PlanarDiagramLinearOperator(PlanarDiagram, LinearOperator):

    def __init__(self, *op_tensors: Tensor):
        self.op_tensors = op_tensors

    @classmethod
    @abstractmethod
    def define_matvec(cls, op: TensorPlaceholder) -> TensorPlaceholder:
        """TODO"""
        ...

    @classmethod
    def get_matvec_contraction_order(cls, tensors, placeholders):
        return PlanarDiagram.get_contraction_order(cls, tensors, placeholders)

    def matvec(self, vec):
        return self.evaluate_matvec(*self.op_tensors, vec)

    @classmethod
    def evaluate_matvec(cls, *op_tensors: Tensor, vec: Tensor):
        tensors = [*op_tensors, vec]
        op_placeholders = cls.define_diagram()
        op_ph = cls.contract_placeholders(op_placeholders)  # FIXME in PlanarDiagram
        vec_ph = cls.define_matvec(op_ph)
        # FIXME stitch contractions of vec_ph to the op_placeholders
        placeholders = [*op_placeholders, vec_ph]
        order = cls.get_matvec_contraction_order(tensors=tensors, placeholders=placeholders)
        return contract(tensors=tensors, placeholders=placeholders, order=order)
    
    ...  # TODO



def contract(tensors: list[Tensor], placeholders: list[Tensor], order: list[int]
             ) -> Tensor | float | complex:
    check_diagram(placeholders)
    if len(tensors) != len(placeholders):
        msg = f'Expected {len(placeholders)} tensors. Got {len(tensors)}.'
        raise ValueError(msg)
    for p, t in zip(placeholders, tensors):
        p.check_compatible_with(t)
    tensors, placeholders = do_traces(tensors, placeholders)
    return do_binary_contractions(tensors, placeholders, order)


def check_diagram(placeholders: list[TensorPlaceholder]):
    """Check if the diagram is well defined. Raise on violations.

    Parameters
    ----------
    placeholders : list of TensorPlaceholder
        The output of :meth:`define_diagram`.
    """
    if len(placeholders) == 0:
        raise ValueError('Diagram is empty')
    open_labels = []
    for p in placeholders:
        if not p.fully_specified:
            raise ValueError('Connectivity of legs is not fully specified!')
        for label, leg in p.contracted_legs.items():
            expect_same = leg.parent.contracted_legs[leg.label]
            if expect_same.parent is not p:
                raise RuntimeError('Inconsistent contraction')
            if expect_same.label != label:
                raise RuntimeError('Inconsistent contraction')
        open_labels.extend(p.open_legs.values())
    if duplicate_entries(open_labels):
        raise ValueError('Open leg labels must be unique')


def do_binary_contractions(tensors: list[Tensor], placeholders: list[TensorPlaceholder],
                            order: list[int]) -> Tensor | float | complex:
    """Perform binary contractions.

    TODO inputs outputs
    """
    # FIXME how to handle labels of open legs?
    res = tensors[order[0]]
    res_ph = placeholders[order[0]]

    for n in order[1:]:
        res_ph, legs1, legs2 = res_ph.binary_contraction(placeholders[n])  # FIXME
        # FIXME other contraction routine that doesnt braid and does left bends instead, as needed!
        res = tdot(res, tensors[n], legs1=legs1, legs2=legs2)

    return res

@staticmethod
def do_traces(tensors: list[Tensor], placeholders: list[TensorPlaceholder]
                ) -> tuple[list[Tensor], list[TensorPlaceholder]]:
    """Perform (partial) traces on all single tensors."""
    raise NotImplementedError  # FIXME


# EXAMPLES
# these would live in tenpy or somewhere else


class DensityMatrixMixingLeft(PlanarDiagram):
    r"""

        |    .---theta----.
        |    |   |    \   |
        |   LP---W0-.  \  |
        |    |   |   \  | |
        |          mixL | |
        |    |   |   /  | |
        |   LP*--W0*-  /  |
        |    |   |    /   |
        |    .---theta*---.
    """

    @classmethod
    def define_diagram(cls):
        Lp = TensorPlaceholder(['vR', 'wR', 'vR*'], dims=['chi', 'w', 'chi'])
        Lp_hc = Lp.dagger()
        W = TensorPlaceholder(['wL', 'p', 'wR', 'p*'], dims=['w', 'd', 'w', 'd'])
        W_hc = W.dagger()
        mix_L = TensorPlaceholder(['wL', 'wL*'], dims=['w', 'w'])
        theta = TensorPlaceholder(['vL', 'p0', 'p1', 'vR'], dims=['chi', 'd', 'd', 'chi'])
        theta_hc = theta.dagger()

        Lp['vR'] @ theta['vL']
        Lp['wR'] @ W['wL']
        Lp['vR*'] @ 'vL'
        theta['p0'] @ W['p*']
        theta['p1'] @ theta_hc['p1*']
        theta['vR'] @ theta_hc['vR*']
        W['p'] @ 'p'
        W['wR'] @ mix_L['wL']
        Lp_hc['vR'] @ 'vL*'
        Lp_hc['wR*'] @ W_hc['wL*']
        Lp_hc['vR*'] @ theta_hc['vL*']
        W_hc['p'] @ theta_hc['p0*']
        W_hc['p*'] @ 'p*'
        W_hc['wR*'] @ mix_L['wL*']

        return [theta, Lp, W, mix_L, theta_hc, Lp_hc, W_hc]

    @classmethod
    def evaluate(cls, theta, Lp, W, mix_L, theta_hc=None, Lp_hc=None, W_hc=None):
        if theta_hc is None:
            theta_hc = theta.dagger()
        if Lp_hc is None:
            Lp_hc = Lp.dagger()
        if W_hc is None:
            W_hc = W.dagger()
        tensors = (theta, Lp, W, mix_L, theta_hc, Lp_hc, W_hc)
        return PlanarDiagram.evaluate(cls, *tensors)


class TwoSiteEffectiveH(PlanarDiagramLinearOperator):
    """Effective Hamiltonian during Two-site DMRG

    The operator is given by the following network::

        |        .---       ---.
        |        |    |   |    |
        |       LP----W0--W1---RP
        |        |    |   |    |
        |        .---       ---.

    and acts on two-site wavefunctions ``theta`` as::

        |        .---       ---.
        |        |    |   |    |
        |       LP----W0--W1---RP
        |        |    |   |    |
        |        .--- theta ---.
    """

    def __init__(self, Lp, W0, W1, Rp):
        # this thin wrapper exists only to give variable names to the arguments
        super().__init__(Lp, W0, W1, Rp)

    def matvec(self, theta):
        # this thin wrapper exists only to give variable names to the arguments
        return super().matvec(vec=theta)

    @classmethod
    def evaluate(cls, Lp, W0, W1, Rp):
        # this thin wrapper exists only to give variable names to the arguments
        return super().evaluate(Lp, W0, W1, Rp)

    @classmethod
    def define_diagram(cls):
        # define the diagram of the operator only, without the vector
        Lp = TensorPlaceholder(['vR', 'wR', 'vR*'], dims=['chi', 'w', 'chi'])
        W0 = TensorPlaceholder(['wL', 'p', 'wR', 'p*'], dims=['w', 'd', 'w', 'd'])
        W1 = W0.copy()
        Rp = TensorPlaceholder(['vL', 'vL*', 'wL'], dims=['chi', 'chi', 'w'])
        Lp['vR'] @ 'vR'
        Lp['wR'] @ W0['wL']
        Lp['vR*'] @ 'vR*'
        W0['p'] @ 'p0'
        W0['p*'] @ 'p0*'
        W0['wR'] @ W1['wL']
        W1['p'] @ 'p1'
        W1['p*'] @ 'p*'
        W1['wR'] @ Rp['wL']
        Rp['vL'] @ 'vL'
        Rp['vL*'] @ 'vL*'
        return [Lp, W0, W1, Rp]

    @classmethod
    def get_contraction_order(cls, tensors, placeholders):
        # overriding this method lets us choose how the diagram comes up with the contraction order
        # here, we can work out on paper that there really is only one choice and hard code it
        # options:
        #   a.  hard-code like here, because we can work it out on paper
        #   b.  greedy optimizer on the fly (default implementation?)
        #   c.  exhaustive search. needs either example values for the dimensions or relations
        #       like ``d^2 < chi`` and should happen "offline" / at development time and be hard-coded
        return [0, 1, 2, 3]  # Lp, W0, W1, Rp

    @classmethod
    def define_matvec(cls, H_eff: TensorPlaceholder):
        # define the contractions of the "operator part" with the vector
        theta = TensorPlaceholder(['vL', 'vR', 'p1', 'p0'])
        H_eff['vL'] @ theta['vR']
        H_eff['p0*'] @ theta['p0']
        H_eff['p1*'] @ theta['p1']
        H_eff['vR'] @ theta['vL']
        H_eff['vL*'] @ 'vR'
        H_eff['vR*'] @ 'vL'
        H_eff['p0'] @ 'p0'
        H_eff['p1'] @ 'p1'
        return theta

    @classmethod
    def define_matvec_order(cls):
        # Lp, W0, theta, W1, Rp
        return [0, 1, -1, 2, 3]
