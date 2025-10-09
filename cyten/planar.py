"""TODO"""
# Copyright (C) TeNPy Developers, Apache license

from abc import ABCMeta, abstractmethod
from typing import Sequence

from .tensors import Tensor
from .sparse import LinearOperator


# TODO list
# - revise class names?
# - could do some metaclass magic to make some of the code (diagram definition etc)
#   run already when the class is *defined*, e.g. at import of tenpy, before any instance
#   is made


class TensorPlaceholder:
    """TODO"""

    def __init__(self, labels: Sequence[str], dims: Sequence[str] = None):
        self.labels = labels  # expected labels in the tensor
        self.dims = dims  # symbolds to express scaling of costs, e.g. 'chi', 'd'
        self.contractions = []  # TODO format (depends on syntax choice?)
        # TODO support without labels as well??


class PlanarDiagram(metaclass=ABCMeta):
    """TODO"""

    def __init__(self, tensors: Sequence[Tensor]):
        self.tensors = tensors
        # TODO might be useful eg for error messages, debugging, visualizint
        #      to know the names of the tensors?
        #      e.g. tensor_names = ['Lp', 'W0', 'W1', 'Rp']

    @classmethod
    @abstractmethod
    def define_diagram(cls):
        """TODO"""
        ...

    @classmethod
    def optimal_order(cls, dims: dict[str, int] = None):
        """TODO"""
        # override this to hard-code an optimal order for contraction
        return None

    def to_tensor(self):
        """TODO"""
        order = self.optimal_order()
        if order is None:
            if any(T.num_parameters > 10_000 for T in self.tensors):
                print('warning: default contraction order is probably inefficient')  # TODO proper warning
            order = range(self.num_contractions)
        for i in order:
            ...


# TODO examples below (these would live in tenpy, not here):


class ApplyTwoSiteEffectiveH(PlanarDiagram):
    """Example use-case for a planar diagram

    The effective two-site Hamiltonian looks like this::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .--- theta ---.
    """
    def __init__(self, Lp, W0, W1, Rp, theta):
        PlanarDiagram.__init__(self, tensors=[Lp, W0, W1, Rp, theta])

    @classmethod
    def define_diagram(cls):
        Lp = TensorPlaceholder(['vR', 'wR', 'vR*'], ['chi', 'w', 'chi'])
        W0 = TensorPlaceholder(['wL', 'p', 'wR', 'p*'], ['w', 'd', 'w', 'd'])
        W1 = W0.copy()
        Rp = TensorPlaceholder(['vL', 'wL', 'vL*'], ['chi', 'w', 'chi'])
        theta = TensorPlaceholder(['vL', 'p0', 'p1', 'vR'], ['chi', 'd', 'd', 'chi'])
        ...  # TODO define contraction, see other example below
        return [Lp, W0, W1, Rp, theta]


class TwoSiteEffectiveH(PlanarDiagram, LinearOperator):
    """Example use-case for a planar diagram

    The effective two-site Hamiltonian looks like this::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.
    """
    def __init__(self, Lp, W0, W1, Rp):
        PlanarDiagram.__init__(self, tensors=[Lp, W0, W1, Rp])

    def matvec(self, theta):
        diagram = ApplyTwoSiteEffectiveH(*self.tensors, theta=theta)
        return diagram.evaluate()

    @classmethod
    def define_diagram(cls, syntax_option='a'):
        Lp = TensorPlaceholder(['vR', 'wR', 'vR*'], ['chi', 'w', 'chi'])
        W0 = TensorPlaceholder(['wL', 'p', 'wR', 'p*'], ['w', 'd', 'w', 'd'])
        W1 = W0.copy()
        Rp = TensorPlaceholder(['vL', 'wL', 'vL*'], ['chi', 'w', 'chi'])

        # TODO how to make the contractions clear. good syntax?
        #      the goal is to modify the placeholders such that they know their contractions
        if syntax_option == 'a':
            cls.define_contraction(Lp['wR'], W0['wL'])
            cls.define_contraction(W0['wR'], W1['wL'])
            cls.define_contraction(W1['wR'], Rp['wL'])

        if syntax_option == 'b':
            cls.define_contraction(Lp, 'wR', W0, 'wL')
            ...

        if syntax_option == 'c':
            Lp.define_contraction('wR', W0, 'wL')
            ...

        if syntax_option == 'd':
            Lp['wR'].contract(W0['wL'])
            ...

        if syntax_option == 'e':
            Lp['wR'] ^ W0['wL']
            ...

        return [Lp, W0, W1, Rp]
