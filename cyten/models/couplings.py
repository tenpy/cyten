"""Couplings are the building blocks of Hamiltonians for lattice models.

This module defines a base class for couplings, which are given in a MPO-like factorized form,
as well as functions that create common couplings such as e.g. a Heisenberg couplings between
two sites that have a spin degree of freedom.
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np

from ..dtypes import Dtype
from ..backends import TensorBackend
from ..backends.abstract_backend import Block
from ..tensors import SymmetricTensor, squeeze_legs, tdot, add_trivial_leg
from .sites import Site, SpinfulSite, GoldenSite


class Coupling:
    """A coupling is a (usually hermitian) operator on a few :class:`Site` s.

    TODO elaborate, use case, examples, maybe look at the docs in tenpy v1 of terms or add_coupling

    Attributes
    ----------
    sites : list of :class:`Site`
        The sites that the operators act on.
    factorization : list of :class:`SymmetricTensor`
        A list of tensors that, if contracted, give the operator that is represented.
        Each tensor ``factorization[i]`` has legs ``[vL, p, vR, p*]``, where ``p`` and ``p*`` are
        the physical :attr:`Site.leg` of the corresponding ``sites[i]``, and where contracting
        the ``vL`` and ``vR`` legs in an MPO-like geometry gives the multi-site operator.
        TODO should we rename vL/vR to wL/wR to match tenpy MPO convention?
    name : str, optional
        A descriptive name that can be used when pretty-printing, to identify the coupling.
        For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
    """

    def __init__(self, sites: list[Site], factorization: list[SymmetricTensor],
                 name: str = None):
        self.sites = sites
        assert len(factorization) == len(sites)
        self.factorization = factorization
        self.name = name
        self.test_sanity()  # OPTIMIZE

    def test_sanity(self):
        for s, W in zip(self.sites, self.factorization):
            s.test_sanity()
            W.test_sanity()
            assert W.num_codomain_legs == 2
            assert W.num_domain_legs == 2
            assert W.labels == ['vL', 'p', 'vR', 'p*']
            assert W.get_leg_co_domain('p') == s.leg
            assert W.get_leg_co_domain('p*') == s.leg
        assert self.factorization[0].get_leg('vL').is_trivial()
        for W1, W2 in zip(self.factorization[:-1], self.factorization[1:]):
            assert W1.get_leg_co_domain('vR') == W2.get_leg_co_domain('vL')
        assert self.factorization[-1].get_leg('vR').is_trivial()

    @classmethod
    def from_dense_block(cls, operator: Block, sites: list[Site], name: str = None,
                         backend: TensorBackend = None, device: str = None,
                         dtype: Dtype = None):
        """TODO elaborate. expect leg order [p0, p1, ..., p0*, p1*, ...]"""
        co_domain = [s.leg for s in sites]
        p_labels = [f'p{i}' for i in range(len(sites))]
        labels = [p_labels, [f'{pi}*' for pi in p_labels]]
        op = SymmetricTensor.from_dense_block(operator, co_domain, co_domain, backend=backend,
                                              labels=labels, dtype=dtype, device=device)
        return cls.from_tensor(op, sites=sites, name=name)

    @classmethod
    def from_tensor(cls, operator: SymmetricTensor, sites: list[Site], name: str = None):
        if len(sites) == 1:
            return OnSiteOperator.from_tensor(operator, sites, name=name)
        raise NotImplementedError  # TODO

    def to_tensor(self) -> SymmetricTensor:
        res = squeeze_legs(self.factorization[0], 'vL')
        for W in self.factorization[1:]:
            res = tdot(res, W, 'vR', 'vL')
        res = squeeze_legs(res, 'vR')
        return res

    def to_numpy(self) -> np.ndarray:
        return self.to_tensor().to_numpy()


class OnSiteOperator(Coupling):

    def __init__(self, site: Site, operator: SymmetricTensor, name: str = None):
        self.operator = operator
        W = add_trivial_leg(operator, domain_pos=0, label='vL')
        W = add_trivial_leg(W, codomain_pos=1, label='vR')
        Coupling.__init__(self, sites=[site], factorization=[W], name=name)

    @classmethod
    def from_tensor(cls, operator: SymmetricTensor, sites: list[Site], name: str = None):
        assert len(sites) == 1
        return cls(site=sites[0], operator=operator, name=name)


def spin_spin_coupling(sites: list[SpinfulSite],
                       xx: float = None, yy: float = None, zz: float = None,
                       backend: TensorBackend = None, device: str = None, name: str = 'spin-spin'
                       ) -> Coupling:
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
    s2 = sites[0].spin_vector
    h = 0  # build in leg order [p0, p0*, p1, p1*] and transpose only once before returning
    if xx is not None:
        h += xx * np.tensordot(s1[:, :, 0], s2[:, :, 0], (0, 0))
    if yy is not None:
        h += yy * np.tensordot(s1[:, :, 0], s2[:, :, 0], (0, 0))
    if zz is not None:
        h += zz * np.tensordot(s1[:, :, 0], s2[:, :, 0], (0, 0))
    if np.ndim(h) == 0:
        raise ValueError('Must have at least one non-zero prefactor.')
    h = np.transpose(h, [0, 2, 1, 3])
    return Coupling.from_dense_block(h, sites, name=name, backend=backend, device=device)


def heisenberg_coupling(sites: list[SpinfulSite], backend: TensorBackend = None, device: str = None,
                        name: str = 'S.S') -> Coupling:
    # TODO test that this builds what we expect
    return spin_spin_coupling(sites=sites, xx=1, yy=1, zz=1, backend=backend, device=device,
                              name=name)


def chiral_3spin_coupling(sites: list[SpinfulSite], backend: TensorBackend = None,
                          device: str = None, name: str = 'S.SxS') -> Coupling:
    # TODO test that this builds what we expect
    assert len(sites) == 3
    SxS = np.cross(sites[1].spin_vector[:, None, :, None],
                   sites[2].spin_vector[None, :, None, :],
                   axis=4)  # [p1, p2, p1*, p2*, i]
    h = np.dot(sites[0].spin_vector, SxS, (0, 0))  # [p0, p0*, p1, p2, p1*, p2*]
    h = np.transpose(h, [0, 2, 3, 1, 4, 5])
    return Coupling.from_dense_block(h, sites, name=name, backend=backend, device=device)


def gold_coupling(sites: list[GoldenSite], backend: TensorBackend = None,
                  device: str = None, name: str = 'P_tau') -> Coupling:
    assert len(sites) == 2  # TODO or should we allow this to generalize???
    tau = [1]
    p_labels = [f'p{i}' for i in range(len(sites))]
    labels = [p_labels, [f'{pi}*' for pi in p_labels]]
    tau_projector = SymmetricTensor.from_sector_projection(
        co_domain=[s.leg for s in sites], sector=tau, backend=backend, labels=labels, device=device
    )
    return Coupling.from_tensor(tau_projector, sites=sites, name=name)


# TODO implement more of these functions that generate couplings. at least cover everything
#      that occurs as a coupling in tenpy v1. also cover some anyon models.
#      if it is a "common" coupling, put it in cyten, if it is "obscure" or very specific,
#      put it in tenpy (for now, we keep a tenpy_models mockup in cyten)
# TODO update cyten/__init__.py and cyten/models/__init__.py accordingly!
