"""Toy code implementing a site."""
# Copyright (C) TeNPy Developers, Apache license
from abc import abstractmethod
import numpy as np
import cyten as ct


class SimpleSite:
    """Simple class for a site.

    By convention, all operators are given the labels ``(vL, p, vR*, p*)``,
    where the leg to the left, `vL`, is part of the codomain and the leg to the
    right, `vR*`, part of the domain.

    Parameters
    ----------
    physical_space : ElementarySpace
        Describes the on-site (physical) Hilbert space.
    backend : TensorBackend
        Backend used for the on-site operators.
    state_labels : None | list[str | None]
        Optional labels for each local basis states. ``None`` entries are
        ignored / not set.
    **site_ops :
        Additional keyword arguments of the form ``name = op`` given to
        :meth:`add_operator`.
    """
    def __init__(self, physical_space: ct.ElementarySpace, backend: ct.TensorBackend | None = None,
                 state_labels: None | list[str | None] = None, **site_ops):
        self.physical_space = physical_space
        self.symmetry = physical_space.symmetry
        if backend is None:
            backend = ct.get_backend(symmetry=self.symmetry)
        self.backend = backend
        self.state_labels = dict()
        if state_labels is not None:
            for i, v in enumerate(state_labels):
                if v is not None:
                    self.state_labels[str(v)] = i
        self.opnames = set()
        for name, op in site_ops.items():
            self.add_operator(name, op)

    def add_operator(self, name: str, op: ct.SymmetricTensor):
        """Add an on-site operator.

        Parameters
        ----------
        name : str
            The variable name used to label the operator. `op` is added as
            attribute to self under this name.
        op : SymmetricTensor
            Tensor representing an on-site operator. Must have legs
            corresponding to the labels ``(vL, p, vR*, p*)``, i.e.,
            ``op.codomain[1] == self.physical_space`` and
            ``op.domain[0] == self.physical_space``.
        """
        assert isinstance(op, ct.SymmetricTensor)
        if name in self.opnames:
            raise ValueError("Operator with that name already existent: " + name)
        if hasattr(self, name):
            raise ValueError("Site already has that attribute name: " + name)
        if not op.backend == self.backend:
            op = ct.SymmetricTensor(data=op.data, codomain=op.codomain,
                                    domain=op.domain, backend=self.backend)

        # TODO maybe we do want to allow inconsistent leg numbers among
        # operators from omitting trivial legs?
        assert op.num_codomain_legs == 2, "Operator has too many legs in the codomain"
        assert op.codomain[1] == self.physical_space
        assert op.num_domain_legs == 2, "Operator has too many legs in the domain"
        assert op.domain[0] == self.physical_space
        
        # TODO do we want to allow different labels?
        # rename labels to fit the convention (vL, p, vR*, p*)
        op.labels = ['vL', 'p', 'vR*', 'p*']
        setattr(self, name, op)
        self.opnames.add(name)

    def add_operators_from_multi_site_op(self, names: list[str | None],
                                         multi_site_op: ct.SymmetricTensor) -> list[ct.SymmetricTensor]:
        """Split an operator into multiple on-site operators and add them.

        It is possible to add on-site operators originating from multi-site
        operators acting on sites with different Hilbert spaces by setting the
        entries in ``names`` corresponding to sites inconsistent with self to
        ``None``.
        
        Parameters
        ----------
        names : list[str | None]
            Names under which the on-site operators obtained from decomposing
            `op` are added to self. ``None`` entries are omitted / not set.
        multi_site_op : SymmetricTensor
            Multi-site operator to be split into on-site operators. The on-site
            operator acting on the `i`-th site is added under the name
            ``names[i]`` to self.

        See Also
        --------
        add_operators_from_dense_multi_site_op
        """
        assert multi_site_op.num_codomain_legs == len(names)
        ops = self.decompose_multi_site_operator(multi_site_op)
        for name, op in zip(names, ops):
            if name is None:
                continue
            self.add_operator(name, op)
        return ops

    def add_operators_from_dense_multi_site_op(self, names: list[str | None],
                                               co_dom: ct.TensorProduct, block: ct.Block,
                                               dtype: ct.Dtype = None) -> list[ct.SymmetricTensor]:
        """Split a dense operator into multiple on-site operators and add them.

        It is possible to add on-site operators originating from multi-site
        operators acting on sites with different Hilbert spaces by setting the
        entries in ``names`` corresponding to sites inconsistent with self to
        ``None``.
        
        Parameters
        ----------
        names : list[str | None]
            Names under which the on-site operators obtained from decomposing
            `op` are added to self. ``None`` entries are omitted / not set.
        co_dom : TensorProduct
            Tensorproduct describing both the codomain and the domain on which
            the multi-site operator acts.
        block : Block
            The data to be converted to the multi-site operator.
        dtype: Dtype
            If given, resulting multi-site operator will have that dtype.
        
        See Also
        --------
        add_operators_from_multi_site_op
        """
        assert len(co_dom) == len(names)
        multi_site_op = ct.SymmetricTensor.from_dense_block(block, co_dom, co_dom,
                                                            backend=self.backend, dtype=dtype)
        return self.add_operators_from_multi_site_op(names, multi_site_op)

    def add_trivial_legs_to_on_site_op(self, op: ct.SymmetricTensor, left: bool,
                                       right: bool) -> ct.SymmetricTensor:
        """Add trivial legs on the left and / or right of an on-site operator.
        
        See Also
        --------
        remove_trivial_legs_from_on_site_op
        """
        trivial_space = ct.ElementarySpace.from_trivial_sector(dim=1, symmetry=self.symmetry)
        idx = op.num_codomain_legs
        labels = op.labels
        data = op.data
        if left:
            assert op.num_codomain_legs == 1
            idx += 1
            labels.insert(0, 'vL')
            new_codom = ct.TensorProduct([trivial_space, op.codomain[0]], symmetry=self.symmetry,
                                         _sector_decomposition=op.codomain.sector_decomposition,
                                         _multiplicities=op.codomain.multiplicities)
            if isinstance(self.backend, ct.AbelianBackend):
                data.block_inds = np.insert(data.block_inds, 0, values=0, axis=1)
                shapes = [self.backend.block_backend.get_shape(block) for block in data.blocks]
                shapes = [[1, *shape] for shape in shapes]
                data.blocks = [self.backend.block_backend.reshape(block, shape)
                               for block, shape in zip(data.blocks, shapes)]
        else:
            new_codom = op.codomain
        if right:
            assert op.num_domain_legs == 1
            labels.insert(idx, 'vR*')
            new_dom = ct.TensorProduct([op.domain[0], trivial_space], symmetry=self.symmetry,
                                       _sector_decomposition=op.domain.sector_decomposition,
                                       _multiplicities=op.domain.multiplicities)
            if isinstance(self.backend, ct.AbelianBackend):
                data.block_inds = np.insert(data.block_inds, idx, values=0, axis=1)
                shapes = [self.backend.block_backend.get_shape(block) for block in data.blocks]
                shapes = [[*shape[:idx], 1, *shape[idx:]] for shape in shapes]
                data.blocks = [self.backend.block_backend.reshape(block, shape)
                               for block, shape in zip(data.blocks, shapes)]
        else:
            new_dom = op.domain
        return ct.SymmetricTensor(data, new_codom, new_dom, self.backend, labels=labels)

    def remove_trivial_legs_from_on_site_op(self, op: ct.SymmetricTensor, left: bool,
                                            right: bool) -> ct.SymmetricTensor:
        """Remove trivial legs on the left and / or right of an on-site operator.
        
        See Also
        --------
        add_trivial_legs_to_on_site_op
        """
        labels = op.labels
        if left:
            assert op.num_codomain_legs == 2
            assert op.codomain[0].is_trivial
            labels = labels[1:]
            new_codom = ct.TensorProduct([op.codomain[1]], symmetry=self.symmetry,
                                         _sector_decomposition=op.codomain.sector_decomposition,
                                         _multiplicities=op.codomain.multiplicities)
        else:
            new_codom = op.codomain
        if right:
            assert op.num_domain_legs == 2
            labels = labels[:op.num_codomain_legs] + labels[op.num_codomain_legs + 1:]
            new_dom = ct.TensorProduct([op.domain[0]], symmetry=self.symmetry,
                                       _sector_decomposition=op.domain.sector_decomposition,
                                       _multiplicities=op.domain.multiplicities)
        else:
            new_dom = op.domain
        return ct.SymmetricTensor(op.data, new_codom, new_dom, self.backend, labels=labels)

    def decompose_multi_site_operator(self, op: ct.SymmetricTensor, add_trivial_leg_left: bool = True,
                                      add_trivial_leg_right: bool = True) -> list[ct.SymmetricTensor]:
        """Decompose a multi-site operator into multiple on-site operators.

        The decomposition is done from right to left using LQ decompositions.
        The additional legs connecting the on-site operators get labels `vL`
        and `vR*`.

        It is not checked whether or not ``op`` has legs that are consistent
        with the physical Hilbert space defined by self. This allows for the
        decomposition multi-site operators acting on sites of different local
        Hilbert spaces.

        Parameters
        ----------
        op : SymmetricTensor
            Multi-site operator to be split into on-site operators.
        add_trivial_leg_left : bool
            Whether or not to add a trivial leg on the left of the left-most
            on-site operator obtained from the decomposition. Doing so results
            in a four-leg tensor that can be added to self using
            :meth:`add_operator`.
        add_trivial_leg_right : bool
            Whether or not to add a trivial leg on the right of the right-most
            on-site operator obtained from the decomposition. Doing so results
            in a four-leg tensor that can be added to self using
            :meth:`add_operator`.
        """
        # do not test that every leg == self.physical leg to allow construction of on-site
        # operators from operators acting on sites with different Hilbert spaces
        assert self.symmetry.is_same_symmetry(op.symmetry)
        assert op.codomain == op.domain

        op_list = []
        # decompose from right to left using LQ decomposition
        # convention for levels: legs to the right have higher levels
        n = op.num_codomain_legs
        levels = [2 * i for i in range(n)]
        levels.extend([2 * i + 1 for i in range(n)][::-1])
        l = ct.permute_legs(op, domain=[n, n - 1], levels=levels)
        l, q = ct.lq(l, ['vR*', 'vL'])        
        q = ct.permute_legs(q, codomain=[0, 1])
        if add_trivial_leg_right:
            q = self.add_trivial_legs_to_on_site_op(q, left=False, right=True)
        op_list.append(q)

        for n in range(op.num_codomain_legs - 1, 1, -1):
            levels = [2 * i for i in range(n)]
            levels.extend([2 * i + 1 for i in range(n)][::-1])
            # for the leg connecting to the previous operator that is already in op_list
            levels.append(2 * n)
            l = ct.permute_legs(l, domain=[n, -1, n - 1], levels=levels)
            l, q = ct.lq(l, ['vR*', 'vL'])
            q = ct.permute_legs(q, codomain=[0, 1])
            op_list.append(q)

        l = ct.permute_legs(l, domain=[1, 2], levels=[0, 1, 2])
        if add_trivial_leg_left:
            l = self.add_trivial_legs_to_on_site_op(l, left=True, right=False)
        op_list.append(l)
        return op_list[::-1]

    def identity_on_site_operator(self, v: ct.ElementarySpace,
                                  overbraid: bool = True) -> ct.SymmetricTensor:
        """Construct on-site identity operator for a given space from the left.

        The identity operator corresponds to ``v`` passing in front of or
        behind the physical Hilbert space.
        
        Parameters
        ----------
        v : ElementarySpace
            Space passing the physical Hilbert space.
        overbraid : bool
            Whether or not ``v`` is in front of the on-site space. This is of
            importance only for symmetries with non-symmetric braids.
        """
        assert self.symmetry.is_same_symmetry(v.symmetry)
        op = ct.SymmetricTensor.from_eye([v, self.physical_space], labels=['vL', 'p', 'p*', 'vR*'])
        levels = list(range(4)) if overbraid else list(range(3, -1, -1))
        op = ct.permute_legs(op, domain=['p*', 'vR*'], levels=levels)
        return op


class SimpleSpinSite(SimpleSite):
    def __init__(self, spin, symmetry, backend = None):
        if isinstance(symmetry, ct.SU2Symmetry):
            sectors = np.array([[spin]])
        elif isinstance(symmetry, ct.U1Symmetry):
            sectors = np.arange(-1 * spin, spin + 2, 2)[:, None]
        else:
            raise NotImplementedError
        physical_space = ct.ElementarySpace(symmetry, sectors)
        super().__init__(physical_space, backend)
        self.add_on_site_spin_ops()

    @abstractmethod
    def add_on_site_spin_ops(self):
        ...


class SimpleU1SpinSite(SimpleSpinSite):
    def __init__(self, spin, backend=None):
        super().__init__(spin, ct.u1_symmetry, backend)

    def add_on_site_spin_ops(self):
        # TODO add sx, sy, sz
        n = self.physical_space.dim
        spin = (n - 1) / 2
        sz = np.diag(-1 * spin + np.arange(n))
        sp = np.zeros([n, n])
        for i in np.arange(n - 1):
            # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
            m = i - spin
            sp[i + 1, i] = np.sqrt(spin * (spin + 1) - m * (m + 1))
        sm = np.transpose(sp)
        # Sp = Sx + i Sy, Sm = Sx - i Sy
        sx = (sp + sm) * 0.5
        sy = (sm - sp) * 0.5j

        slist = [sx, sy, sz]
        # construct dense operator corresponding to s . s
        s_dot_s_dense = np.zeros([n] * 4, dtype=complex)
        for si in slist:
            s_dot_s_dense += si[:, None, None, :] * si[None, :, :, None]
        co_dom = ct.TensorProduct([self.physical_space] * 2, self.symmetry)
        names = ['s_dot', 'dot_s']
        self.add_operators_from_dense_multi_site_op(names=names, co_dom=co_dom,
                                                    block=s_dot_s_dense)

        # construct dense operator corresponding to s . s x s
        s_dot_s_x_s_dense = np.zeros([n] * 6, dtype=complex)
        for i in range(3):
            s_dot_s_x_s_dense += (slist[i % 3][:, None, None, None, None, :]
                                  * slist[(i + 1) % 3][None, :, None, None, :, None]
                                  * slist[(i + 2) % 3][None, None, :, :, None, None])
            s_dot_s_x_s_dense -= (slist[i % 3][:, None, None, None, None, :]
                                  * slist[(i + 2) % 3][None, :, None, None, :, None]
                                  * slist[(i + 1) % 3][None, None, :, :, None, None])
        co_dom = ct.TensorProduct([self.physical_space] * 3, self.symmetry)
        names = ['s_dot_x', 'dot_s_x', 'dot_x_s']
        self.add_operators_from_dense_multi_site_op(names=names, co_dom=co_dom,
                                                    block=s_dot_s_x_s_dense)


class SimpleSU2SpinSite(SimpleSpinSite):
    def __init__(self, spin, backend=None):
        super().__init__(spin, ct.SU2Symmetry(), backend)

    def add_on_site_spin_ops(self):
        self.add_s_dot_s_ops()
        self.add_s_dot_s_x_s_ops()

    def add_s_dot_s_ops(self):
        s_dot_s = ct.SymmetricTensor.from_eye([self.physical_space] * 2, backend=self.backend)
        spin = (self.physical_space.dim - 1) / 2
        on_site_casimir = spin * (spin + 1)
        for block, idcs in zip(s_dot_s.data.blocks, s_dot_s.data.block_inds):
            coupled_spin = s_dot_s.domain.sector_decomposition[idcs[1]][0] / 2
            coupled_casimir = coupled_spin * (coupled_spin + 1)
            block[0, 0] = coupled_casimir / 2 - on_site_casimir
        self.add_operators_from_multi_site_op(['s_dot', 'dot_s'], s_dot_s)

    def add_s_dot_s_x_s_ops(self):
        n = self.physical_space.dim
        spin = (n - 1) / 2
        sz = np.diag(-1 * spin + np.arange(n))
        sp = np.zeros([n, n])
        for i in np.arange(n - 1):
            # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
            m = i - spin
            sp[i + 1, i] = np.sqrt(spin * (spin + 1) - m * (m + 1))
        sm = np.transpose(sp)
        # Sp = Sx + i Sy, Sm = Sx - i Sy
        sx = (sp + sm) * 0.5
        sy = (sm - sp) * 0.5j

        # construct dense operator corresponding to s . s x s
        slist = [sx, sy, sz]
        s_dot_s_x_s_dense = 0
        for i in range(3):
            s_dot_s_x_s_dense += (slist[i % 3][:, None, None, None, None, :]
                                  * slist[(i + 1) % 3][None, :, None, None, :, None]
                                  * slist[(i + 2) % 3][None, None, :, :, None, None])
            s_dot_s_x_s_dense -= (slist[i % 3][:, None, None, None, None, :]
                                  * slist[(i + 2) % 3][None, :, None, None, :, None]
                                  * slist[(i + 1) % 3][None, None, :, :, None, None])

        co_dom = ct.TensorProduct([self.physical_space] * 3, self.symmetry)
        names = ['s_dot_x', 'dot_s_x', 'dot_x_s']
        self.add_operators_from_dense_multi_site_op(names=names, co_dom=co_dom,
                                                    block=s_dot_s_x_s_dense)
