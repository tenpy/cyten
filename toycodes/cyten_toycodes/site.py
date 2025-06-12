"""Toy code implementing a site."""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from abc import abstractmethod
import numpy as np
import cyten as ct


class Coupling:
    """Simple class for couplings, i.e., operators on sites.

    By convention, all single-site operators making up a coupling have four
    legs and are given labels ``(vL, pi, vR*, pi*)``, where the leg to the
    left, `vL`, is part of the codomain and the leg to the right, `vR*`, part
    of the domain. The `i` in `pi` is an integer that coincides with the number
    enumerating the attribute ``tensors``.

    Parameters
    ----------
    names : list[str]
        All the (equivalent) names used to refer to the coupling.
    tensors : list[SymmetricTensor]
        Tensors defining the coupling. The codomain and domain of each tensor
        in this list must contain two legs such that
        ``tensors[i].codomain[1] == tensors[i].domain[0]``.
    """

    def __init__(self, names: list[str], tensors: list[ct.SymmetricTensor]):
        self.names = names
        self.symmetry = tensors[0].symmetry
        self.backend = tensors[0].backend
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def test_sanity(self):
        for name in self.names:
            assert isinstance(name, str)
        assert isinstance(self.symmetry, ct.symmetries.Symmetry)
        assert isinstance(self.backend, ct.backends.TensorBackend)
        for i, ten in enumerate(self.tensors):
            assert self.symmetry.is_same_symmetry(ten.symmetry)
            assert ten.backend == self.backend
            assert ten.num_codomain_legs == 2
            assert ten.num_domain_legs == 2
            assert ten.codomain[1] == ten.domain[0]
            ten.test_sanity()
            # make sure the tensors can be contracted
            if i < len(self) - 1:
                assert ten.domain[1] == self.tensors[i + 1].codomain[0]

    @classmethod
    def from_operator(cls, names: list[str], op: ct.SymmetricTensor) -> Coupling:
        """Convert an operator / symmetric tensor to a coupling.

        Parameters
        ----------
        names : list[str]
            All the (equivalent) names used to refer to the coupling.
        op : SymmetricTensor
            Tensor to be converted to a coupling. May act on a single or on
            multiple sites.
        """
        assert op.codomain == op.domain
        labels = [f'p{i}' for i in range(op.num_codomain_legs)]
        labels.extend([l + '*' for l in labels[::-1]])
        op.labels = labels
        if op.num_codomain_legs > 1:
            tensors = decompose_multi_site_operator(op)
        else:
            tensors = [add_trivial_legs_to_on_site_op(op)]
        return cls(names, tensors)

    @classmethod
    def from_dense_block(cls, names: list[str], co_dom: ct.TensorProduct,
                         block: ct.Block, backend: ct.backends.TensorBackend,
                         dtype: ct.Dtype | None = None) -> Coupling:
        """Convert an operator defined via a dense block to a coupling.

        Parameters
        ----------
        names : list[str]
            All the (equivalent) names used to refer to the coupling.
        co_dom : TensorProduct
            Tensorproduct describing both the codomain and the domain on which
            the operator acts.
        block : Block
            The block data to be converted to the operator.
        backend : TensorBackend
            Backend of the coupling.
        dtype: Dtype
            If given, resulting coupling will have that dtype.
        """
        op = ct.SymmetricTensor.from_dense_block(block, codomain=co_dom, domain=co_dom,
                                                 backend=backend, dtype=dtype)
        return cls.from_operator(names, op)

    def to_numpy(self) -> ct.SymmetricTensor:
        """Convert to numpy array."""
        # the convention for the decomposition is that legs to the right have
        # higher levels -> this must now also hold
        op = self.tensors[0]
        for n in range(len(self) - 1):
            levels = list(range(2 * n + 1))
            levels.extend([2 * n + 1, 2 * n + 3, 2 * n + 2])
            op = ct.permute_legs(op, domain=['vR*'])
            op = ct.tdot(op, self.tensors[n + 1], legs1='vR*', legs2='vL')
        op = ct.squeeze_legs(op, legs=['vL', 'vR*'])
        codom_labels = [f'p{i}' for i in range(len(self))]
        dom_labels = [l + '*' for l in codom_labels]
        op = ct.permute_legs(op, codomain=codom_labels, domain=dom_labels,
                             levels=list(range(2 * len(self))))
        return op.to_numpy()
        

class SimpleSite:
    """Simple class for a site.

    Operators acting on the sites are :class:`Coupling`s

    TODO state_labels: should be added to `ElementarySpace` and describe the
    states in the public basis.

    Parameters
    ----------
    physical_space : ElementarySpace
        Describes the on-site (physical) Hilbert space.
    backend : TensorBackend
        Backend used for the operators.
    **couplings : dict[str, Coupling]
        Additional keyword arguments of the form ``name = coupling`` given to
        :meth:`add_coupling`.
    """

    def __init__(self, physical_space: ct.ElementarySpace,
                 backend: ct.backends.TensorBackend | None = None,
                 **couplings: dict[str, Coupling]):
        self.physical_space = physical_space
        self.symmetry = physical_space.symmetry
        if backend is None:
            backend = ct.get_backend(symmetry=self.symmetry)
        self.backend = backend
        self.couplings: dict[str, Coupling] = dict()
        for name, coupling in couplings.items():
            self.add_coupling(name, coupling)

    def test_sanity(self):
        assert isinstance(self.physical_space, ct.ElementarySpace)
        self.physical_space.test_sanity()
        assert isinstance(self.symmetry, ct.symmetries.Symmetry)
        assert isinstance(self.backend, ct.backends.TensorBackend)
        for coupling in self.couplings.values():
            coupling.test_sanity()
            for ten in coupling.tensors:
                assert ten.domain[0] == self.physical_space

    def add_coupling(self, coupling: Coupling):
        """Add a coupling.

        Parameters
        ----------
        coupling : Coupling
            Coupling to be added. Must have consistent backend and legs acting
            on the physical space.

        See Also
        --------
        add_coupling_from_operator, add_coupling_from_dense_block
        """
        assert self.backend == coupling.backend
        for ten in coupling.tensors:
            assert ten.domain[0] == self.physical_space
        for name in coupling.names:
            if name in self.couplings:
                raise ValueError("Coupling with that name already existent: " + name)
            self.couplings[name] = coupling

    def add_coupling_from_operator(self, names: list[str], op: ct.SymmetricTensor):
        """Add a coupling from a tensor.

        Parameters
        ----------
        names : list[str]
            All the (equivalent) names used to refer to the coupling.
        op : SymmetricTensor
            Tensor to be converted to a coupling. May act on a single or on
            multiple sites.

        See Also
        --------
        add_coupling, add_coupling_from_dense_block
        """
        assert self.backend == op.backend
        coupling = Coupling.from_operator(names, op)
        self.add_coupling(coupling)

    def add_coupling_from_dense_block(self, names: list[str], co_dom: ct.TensorProduct,
                                      block: ct.Block, dtype: ct.Dtype | None = None):
        """Add a coupling from a dense block.

        Parameters
        ----------
        names : list[str]
            All the (equivalent) names used to refer to the coupling.
        co_dom : TensorProduct
            Tensorproduct describing both the codomain and the domain on which
            the operator acts.
        block : Block
            The block data to be converted to the operator.
        dtype: Dtype
            If given, resulting coupling will have that dtype.

        See Also
        --------
        add_coupling, add_coupling_from_operator
        """
        coupling = Coupling.from_dense_block(names, co_dom, block, backend=self.backend, dtype=dtype)
        self.add_coupling(coupling)


class SimpleSpinSite(SimpleSite):
    """Simple class for a spin site with SU(2) or U(1) symmetry.

    Parameters
    ----------
    spin : int
        2 * spin on the site.
    symmetry : Symmetry
        Conserved symmetry. Must be SU(2) or U(1) symmetry.
    backend, **couplings: see :class:`SimpleSite`.
    """

    def __init__(self, spin: int, symmetry: ct.symmetries.Symmetry,
                 backend: ct.backends.TensorBackend = None, **couplings):
        assert isinstance(spin, int)
        if symmetry.is_same_symmetry(ct.su2_symmetry):
            sectors = np.array([[spin]])
        elif symmetry.is_same_symmetry(ct.u1_symmetry):
            sectors = np.arange(-1 * spin, spin + 2, 2)[:, None]
        else:
            raise NotImplementedError
        physical_space = ct.ElementarySpace(symmetry, sectors)
        super().__init__(physical_space, backend, **couplings)
        self.add_on_site_spin_ops()

    @abstractmethod
    def add_on_site_spin_ops(self):
        """Add common combinations of spin operators as couplings.

        The added couplings correspond to the SU(2) symmetric multi-site
        operators ``S dot S`` and ``S dot (S x S)`` for U(1) and SU(2)
        symmetry. For U(1) symmetry, the couplings ``Sz``, ``Sz Sz``,
        ``S+ S-`` and ``S- S+`` are also added.
        """
        ...


class SimpleU1SpinSite(SimpleSpinSite):
    """Simple class for a spin site with U(1) symmetry."""

    def __init__(self, spin: int, backend: ct.backends.TensorBackend = None):
        super().__init__(spin, ct.u1_symmetry, backend)

    def add_on_site_spin_ops(self):
        spin = self.physical_space.dim - 1
        sz = np.diag(-1 * spin / 2 + np.arange(self.physical_space.dim))
        co_dom = ct.TensorProduct([self.physical_space], self.symmetry)
        self.add_coupling_from_dense_block(['Sz'], co_dom=co_dom, block=sz)

        dense_blocks = get_dense_spin_operators(spin)
        names = ['Sz Sz', 'S+ S-', 'S- S+', 'S dot S', 'S dot (S x S)']
        co_dom = ct.TensorProduct([self.physical_space] * 2, self.symmetry)
        for block, name in zip(dense_blocks[:4], names[:4]):
            self.add_coupling_from_dense_block([name], co_dom=co_dom, block=block)

        co_dom = ct.TensorProduct([self.physical_space] * 3, self.symmetry)
        self.add_coupling_from_dense_block([names[-1]], co_dom=co_dom,
                                           block=dense_blocks[-1])


class SimpleSU2SpinSite(SimpleSpinSite):
    """Simple class for a spin site with SU(2) symmetry."""

    def __init__(self, spin: int, backend: ct.backends.TensorBackend = None):
        super().__init__(spin, ct.su2_symmetry, backend)

    def add_on_site_spin_ops(self):
        self.add_s_dot_s_ops()
        self.add_s_dot_s_x_s_ops()

    def add_s_dot_s_ops(self):
        """Add the coupling corresponding to ``S dot S``."""
        s_dot_s = ct.SymmetricTensor.from_eye([self.physical_space] * 2,
                                              backend=self.backend)
        spin = (self.physical_space.dim - 1) / 2
        on_site_casimir = spin * (spin + 1)
        for block, idcs in zip(s_dot_s.data.blocks, s_dot_s.data.block_inds):
            coupled_spin = s_dot_s.domain.sector_decomposition[idcs[1]][0] / 2
            coupled_casimir = coupled_spin * (coupled_spin + 1)
            block[0, 0] = coupled_casimir / 2 - on_site_casimir
        self.add_coupling_from_operator(['S dot S'], s_dot_s)

    def add_s_dot_s_x_s_ops(self):
        """Add the couping corresponding to ``S dot (S x S)``."""
        spin = self.physical_space.dim - 1
        dense_block = get_dense_spin_operators(spin)[-1]
        co_dom = ct.TensorProduct([self.physical_space] * 3, self.symmetry)
        self.add_coupling_from_dense_block(['S dot (S x S)'], co_dom=co_dom,
                                           block=dense_block)


def _add_trivial_leg_abelian_data(data: ct.backends.AbelianBackendData,
                                  backend: ct.backends.TensorBackend,
                                  leg_idx: int) -> ct.backends.AbelianBackendData:
    """Change abelian backend data when adding a trivial leg to the associated tensor."""
    data.block_inds = np.insert(data.block_inds, leg_idx, values=0, axis=1)
    shapes = [backend.block_backend.get_shape(block) for block in data.blocks]
    shapes = [[*shape[:leg_idx], 1, *shape[leg_idx:]] for shape in shapes]
    data.blocks = [backend.block_backend.reshape(block, shape)
                   for block, shape in zip(data.blocks, shapes)]
    return data


def add_trivial_legs_to_on_site_op(op: ct.SymmetricTensor) -> ct.SymmetricTensor:
    """Add trivial legs to an on-site operator.
    
    Trivial legs are only added to the co_domain if it contains a single leg.
    For co_domains with two legs, no trivial leg is added.

    Parameters
    ----------
    op : SymmetricTensor
        On-site operator to which trivial legs should be added.
    """
    trivial_space = ct.ElementarySpace.from_trivial_sector(dim=1, symmetry=op.symmetry)
    labels = op.labels
    data = op.data
    backend = op.backend
    if op.num_codomain_legs == 1:
        labels.insert(0, 'vL')
        new_codom = ct.TensorProduct([trivial_space, op.codomain[0]], symmetry=op.symmetry,
                                     _sector_decomposition=op.codomain.sector_decomposition,
                                     _multiplicities=op.codomain.multiplicities)
        if isinstance(backend, ct.backends.AbelianBackend):
            _add_trivial_leg_abelian_data(data, backend, leg_idx=0)
    else:
        assert op.num_codomain_legs == 2
        new_codom = op.codomain

    if op.num_domain_legs == 1:
        labels.insert(2, 'vR*')
        new_dom = ct.TensorProduct([op.domain[0], trivial_space], symmetry=op.symmetry,
                                   _sector_decomposition=op.domain.sector_decomposition,
                                   _multiplicities=op.domain.multiplicities)
        if isinstance(backend, ct.backends.AbelianBackend):
            _add_trivial_leg_abelian_data(data, backend, leg_idx=2)
    else:
        assert op.num_domain_legs == 2
        new_dom = op.domain
    return ct.SymmetricTensor(data, new_codom, new_dom, backend, labels=labels)


def decompose_multi_site_operator(op: ct.SymmetricTensor) -> list[ct.SymmetricTensor]:
    """Decompose a multi-site operator into multiple on-site operators.

    The decomposition is done from right to left using SVDs. The additional
    legs connecting the on-site operators get labels `vL` and `vR*`.

    Parameters
    ----------
    op : SymmetricTensor
        Multi-site operator to be split into on-site operators.
    """
    op_list = []
    # decompose from right to left using SVD
    # convention for levels: legs to the right have higher levels
    n = op.num_codomain_legs
    levels = [2 * i for i in range(n)]
    levels.extend([2 * i + 1 for i in range(n)][::-1])

    u = ct.permute_legs(op, domain=[n, n - 1], levels=levels)
    u, s, v = ct.svd(u, ['vR*', 'vL'])
    u = ct.scale_axis(u, s, leg='vR*')
    v = ct.permute_legs(v, codomain=[0, 1])
    v = add_trivial_legs_to_on_site_op(v)
    op_list.append(v)

    for n in range(op.num_codomain_legs - 1, 1, -1):
        levels = [2 * i for i in range(n)]
        levels.extend([2 * i + 1 for i in range(n)][::-1])
        # for the leg connecting to the previous operator that is already in op_list
        levels.append(2 * n)
        u = ct.permute_legs(u, domain=[n, -1, n - 1], levels=levels)
        u, s, v = ct.svd(u, ['vR*', 'vL'])
        u = ct.scale_axis(u, s, leg='vR*')
        v = ct.permute_legs(v, codomain=[0, 1])
        op_list.append(v)

    u = ct.permute_legs(u, domain=[1, 2], levels=[0, 1, 2])
    u = add_trivial_legs_to_on_site_op(u)
    op_list.append(u)
    return op_list[::-1]


def get_dense_spin_operators(spin: int) -> list[np.ndarray]:
    """Construct symmetric spin operators as dense blocks.

    The constructed dense blocks correspond to the U(1) symmetric operators
    ``Sz Sz``, ``S+ S-`` and ``S- S+``, and to the SU(2) symmetric operators
    ``S . S`` and ``S . (S x S)``.
    
    Parameters
    ----------
    spin : int
        Correspons to 2 * S, with S the spin to be considered.
    """
    n = spin + 1
    spin_ = spin / 2
    sz = np.diag(-1 * spin_ + np.arange(n))
    sp = np.zeros([n, n])
    for i in np.arange(n - 1):
        # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
        m = i - spin_
        sp[i + 1, i] = np.sqrt(spin_ * (spin_ + 1) - m * (m + 1))
    sm = np.transpose(sp)
    # Sp = Sx + i Sy, Sm = Sx - i Sy
    sx = (sp + sm) * 0.5
    sy = (sm - sp) * 0.5j

    szsz = sz[:, None, None, :] * sz[None, :, :, None]
    spsm = sp[:, None, None, :] * sm[None, :, :, None]
    smsp = sm[:, None, None, :] * sp[None, :, :, None]

    slist = [sx, sy, sz]
    # construct dense operator corresponding to s . s
    s_dot_s = np.zeros([n] * 4, dtype=complex)
    for si in slist:
        s_dot_s += si[:, None, None, :] * si[None, :, :, None]

    # construct dense operator corresponding to s . s x s
    s_dot_s_x_s = np.zeros([n] * 6, dtype=complex)
    for i in range(3):
        s_dot_s_x_s += (slist[i % 3][:, None, None, None, None, :] *
                        slist[(i + 1) % 3][None, :, None, None, :, None] *
                        slist[(i + 2) % 3][None, None, :, :, None, None])
        s_dot_s_x_s -= (slist[i % 3][:, None, None, None, None, :] *
                        slist[(i + 2) % 3][None, :, None, None, :, None] *
                        slist[(i + 1) % 3][None, None, :, :, None, None])
    return szsz, spsm, smsp, s_dot_s, s_dot_s_x_s


def verify_operator_decomposition_spins(spin: int = 1):
    """Check that the spin sites work and verify their couplings.
    
    Verifies that the decompositions of the SU(2) symmetric operators ``S . S``
    and ``S . (S x S)`` are correct for both U(1) and SU(2) symmetric sites.
    For U(1) symmetry, the couplings ``Sz``, ``Sz Sz``, ``S+ S-`` and ``S- S+``
    are also checked.
    """
    szsz, spsm, smsp, s_dot_s, s_dot_s_x_s = get_dense_spin_operators(spin)

    np_bak = ct.backends.NumpyBlockBackend()
    siteU1_1 = SimpleU1SpinSite(spin, backend=ct.backends.AbelianBackend(np_bak))
    siteU1_2 = SimpleU1SpinSite(spin, backend=ct.backends.FusionTreeBackend(np_bak))
    siteSU2 = SimpleSU2SpinSite(spin)
    for site in [siteU1_1, siteU1_2, siteSU2]:
        site.test_sanity()

        if ct.u1_symmetry.is_same_symmetry(site.symmetry):
            np.testing.assert_almost_equal(site.couplings['Sz Sz'].to_numpy(), szsz)
            np.testing.assert_almost_equal(site.couplings['S+ S-'].to_numpy(), spsm)
            np.testing.assert_almost_equal(site.couplings['S- S+'].to_numpy(), smsp)

        np.testing.assert_almost_equal(site.couplings['S dot S'].to_numpy(), s_dot_s)
        np.testing.assert_almost_equal(site.couplings['S dot (S x S)'].to_numpy(), s_dot_s_x_s)


if __name__ == '__main__':
    verify_operator_decomposition_spins(spin=1)
