r"""Implements the fusion tree backend.

.. _fusion_tree_backend__blocks:

Blocks
------
A block :math:`T_c` of a symmetric tensor is associated with a coupled sector :math:`c` and is a
matrix. It has the following indices

    [ [ T_c ]^{a_1, ..., a_J, 𝛼}_{b_1, ..., b_K, β} ]^{m_1, ..., m_J}_{n_1, ..., n_K}

Where :math:`a_j` is an uncoupled sector ``a_j = codomain[j].sector_decomposition[i_j]`` of the
a space in the codomain, and ``0 <= m_j < codomain[j].multiplicities[i_j]`` is an associated
multiplicity index, and :math:`𝛼` labels a fusion tree ``(a_1, ..., a_J) -> c``.
Similarly, :math:`b_k` are uncoupled sectors ``b_k = domain[k].sector_decomposition[i_k]``, and
``0 <= n_k < domain[k].multiplicities[i_k]`` is a multiplicity index and :math:`β` labels
a fusion tree ``(b_1, ..., b_K) -> c``.

We call ``T_c`` a *block* and  ``[ T_c ]^{a_1, ..., a_J, 𝛼}_{b_1, ..., b_K, β}`` a *tree block*.
We group the tree blocks with the same uncoupled sectors to a *forest block*
``[ T_c ]^{a_1, ..., a_J}_{b_1, ..., b_K}``.

The blocks parametrize a tensor as::

    |                                                                        V1         VK
    |                                                                         │    │    │
    |                                                                      m1 ▽    ▽    ▽ mJ
    |                                                                      a1 ↑    ↓    ↑ aJ
    |                                                                         │    Z    │
    |   V1     VJ                                                          a1 ↑    ↑    ↑ aJ
    |    ↑  ↓  ↑                                                             ┏┷━━━━┷━━━━┷┓
    |    │  │  │                                                             ┃     𝛼     ┃
    |   ┏┷━━┷━━┷┓                              ┌ ┌     ┐a1..aJ,𝛼 ┐m1..mJ     ┗━━━━━┯━━━━━┛
    |   ┃   T   ┃   =   sum     sum     sum    │ │ T_c │         │                 │ c
    |   ┗┯━━┯━━┯┛      b1..bK  a1..aJ    c     └ └     ┘b1..bK,β ┘n1..nK     ┏━━━━━┷━━━━━┓
    |    │  │  │       n1..nK  m1..mJ   𝛼 β                                  ┃     β     ┃
    |    ↓  ↓  ↑                                                             ┗┯━━━━┯━━━━┯┛
    |   W1     WK                                                          b1 ↑    ↑    ↑ bK
    |                                                                         Z    Z    │
    |                                                                 bar(b1) ↓    ↓    ↑ bK
    |                                                                      n1 △    △    △ nK
    |                                                                         │    │    │
    |                                                                        W1         WK

And we store the blocks as matrices, with combined multi-indices::

    |   ┌ ┌     ┐a1..aJ,𝛼 ┐m1..mJ
    |   │ │ T_c │         │         =   blocks[c_idx][M, N]
    |   └ └     ┘b1..bK,β ┘n1..nK

where ``c = codomain.sector_decomposition[block_inds[c_idx, 0]]`` and
``M = stridify(a1, ..., aJ, 𝛼, m1, ..., mJ)``, i.e. such that ``mJ`` changes the fastest when
``M`` is increased, and analogously ``N = stridify(b1, ..., bK, β, n1, ..., nK)``.
See the following methods for the respective slices / strides of the indices ``M, N``::

    - :meth:`TensorProduct.forest_block_size`
    - :meth:`TensorProduct.forest_block_slice`
    - :meth:`TensorProduct.forest_tree_size`
    - :meth:`TensorProduct.forest_tree_slice`

Visually, the blocks have the following structure::

    |         --------------------------> (b1...bK)
    |          ----->β ------->β ------>β
    |    |    ┏━━━┯━━━┳━┯━┯━┯━┯━┳━━┯━━┯━━┓
    |    | |  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
    |    | |  ┠───┼───╂─┼─┼─┼─┼─╂──┼──┼──┃
    |    | |  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
    |    | v  ┠───┼───╂─┼─┼─┼─┼─╂──┼──┼──┃
    |    | 𝛼  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
    |    |    ┣━━━┿━━━╋━┿━┿━┿━┿━╋━━┿━━┿━━┫
    |    | |  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
    |    | v  ┠───┼───╂─┼─┼─┼─┼─╂──┼──┼──┃
    |    | 𝛼  ┃   │   ┃ │ │ │ │ ┃  │  │  ┃
    |    V    ┗━━━┷━━━┻━┷━┷━┷━┷━┻━━┷━━┷━━┛
    |  (a1..aJ)
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Iterable
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import prod

import numpy as np
import warnings

from .abstract_backend import (
    TensorBackend, BlockBackend, Block, Data, DiagonalData, MaskData,
    conventional_leg_order
)
from ..dtypes import Dtype
from ..symmetries import Sector, Symmetry, BraidChiralityUnspecifiedError
from ..spaces import Space, ElementarySpace, TensorProduct, LegPipe
from ..trees import FusionTree, fusion_trees
from ..tools.misc import (
    inverse_permutation, iter_common_sorted_arrays, iter_common_noncommon_sorted,
    iter_common_sorted, permutation_as_swaps, rank_data
)
from ..tools.mappings import SparseMapping, IdentityMapping

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask


def _tree_block_iter(a: SymmetricTensor):
    sym = a.symmetry
    domain_are_dual = [sp.is_dual for sp in a.domain.factors]
    codomain_are_dual = [sp.is_dual for sp in a.codomain.factors]
    for (bi, _), block in zip(a.data.block_inds, a.data.blocks):
        coupled = a.codomain.sector_decomposition[bi]
        i1_forest = 0  # start row index of the current forest block
        i2_forest = 0  # start column index of the current forest block
        for b_sectors, b_mults in a.domain.iter_uncoupled():
            tree_block_width = np.prod(b_mults)
            forest_block_width = 0
            for a_sectors, a_mults in a.codomain.iter_uncoupled():
                tree_block_height = np.prod(a_mults)
                i1 = i1_forest  # start row index of the current tree block
                i2 = i2_forest  # start column index of the current tree block
                for alpha_tree in fusion_trees(sym, a_sectors, coupled, codomain_are_dual):
                    i2 = i2_forest  # reset to the left of the current forest block
                    for beta_tree in fusion_trees(sym, b_sectors, coupled, domain_are_dual):
                        idx1 = slice(i1, i1 + tree_block_height)
                        idx2 = slice(i2, i2 + tree_block_width)
                        entries = block[idx1, idx2]
                        yield alpha_tree, beta_tree, entries
                        i2 += tree_block_width  # move right by one tree block
                    i1 += tree_block_height  # move down by one tree block
                forest_block_height = i1 - i1_forest
                forest_block_width = max(forest_block_width, i2 - i2_forest)
                i1_forest += forest_block_height
            i1_forest = 0  # reset to the top of the block
            i2_forest += forest_block_width


class FusionTreeData:
    r"""Data stored in a Tensor for :class:`FusionTreeBackend`.

    Attributes
    ----------
    block_inds : 2D array
        Indices that specify the coupled sectors of the non-zero blocks.
        ``block_inds[n] == [i, j]`` indicates that the coupled sector for ``blocks[n]`` is given by
        ``tensor.codomain.sector_decomposition[i] == coupled == tensor.domain.sector_decomposition[j]``.
    blocks : list of 2D Block
        The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
    dtype : Dtype
    device : str
        The device on which the blocks are currently stored.
        We currently only support tensors which have all blocks on a single device.
        Should be the device returned by :func:`BlockBackend.as_device`.
    is_sorted : bool
        If ``False`` (default), we permute `blocks` and `block_inds` according to
        ``np.lexsort(block_inds.T)``.
        If ``True``, we assume they are sorted *without* checking.
    """

    def __init__(self, block_inds: np.ndarray, blocks: list[Block], dtype: Dtype, device: str,
                 is_sorted: bool = False):
        if not is_sorted:
            perm = np.lexsort(block_inds.T)
            block_inds = block_inds[perm, :]
            blocks = [blocks[n] for n in perm]
        self.block_inds = block_inds
        self.blocks = blocks
        self.dtype = dtype
        self.device = device

    def block_ind_from_coupled(self, coupled: Sector, domain: TensorProduct) -> int | None:
        """Return `ind` such that ``blocks[ind]`` is associated with the `coupled` sector.

        This is such that ``domain.sector_decomposition[block_inds[res][1]] == coupled``.

        Note: we use the domain (and not the codomain), since only the :attr:`block_inds[:, 1]``
        are sorted.
        """
        domain_sector_ind = domain.sector_decomposition_where(coupled)
        return self.block_ind_from_domain_sector_ind(domain_sector_ind)

    def block_ind_from_domain_sector_ind(self, domain_sector_ind: int) -> int | None:
        """Return `ind` such that ``block_inds[ind, 1] == domain_sector_ind``

        Note: we use the domain (and not the codomain), since only the :attr:`block_inds[:, 1]``
        are sorted.
        """
        ind = np.searchsorted(self.block_inds[:, 1], domain_sector_ind)
        if ind >= len(self.block_inds) or self.block_inds[ind, 1] != domain_sector_ind:
            return None
        if ind + 1 < self.block_inds.shape[0] and self.block_inds[ind + 1, 1] == domain_sector_ind:
            raise RuntimeError
        return ind

    def discard_zero_blocks(self, backend: BlockBackend, eps: float) -> None:
        """Discard blocks whose norm is below the threshold `eps`"""
        keep = []
        for i, block in enumerate(self.blocks):
            if backend.norm(block) >= eps:
                keep.append(i)
        self.blocks = [self.blocks[i] for i in keep]
        self.block_inds = self.block_inds[keep]

    def save_hdf5(self, hdf5_saver, h5gr, subpath):

        hdf5_saver.save(self.block_inds, subpath + 'block_inds')
        hdf5_saver.save(self.blocks, subpath + 'blocks')
        hdf5_saver.save(self.dtype.to_numpy_dtype(), subpath + 'dtype')
        hdf5_saver.save(self.device, subpath + 'device')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):

        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.block_inds = hdf5_loader.load(subpath + 'block_inds')
        obj.blocks = hdf5_loader.load(subpath + 'blocks')
        obj.device = hdf5_loader.load(subpath + 'device')
        dt = hdf5_loader.load(subpath + 'dtype')
        obj.dtype = Dtype.from_numpy_dtype(dt)

        return obj


class FusionTreeBackend(TensorBackend):
    """A backend based on fusion trees."""

    DataCls = FusionTreeData
    can_decompose_tensors = True

    def __init__(self, block_backend: BlockBackend, eps: float = 1.e-14):
        self.eps = eps
        super().__init__(block_backend)

    def test_tensor_sanity(self, a: SymmetricTensor | DiagonalTensor | Mask, is_diagonal: bool):
        super().test_tensor_sanity(a, is_diagonal=is_diagonal)
        data: FusionTreeData = a.data
        # check device and dtype
        assert a.device == data.device == self.block_backend.as_device(data.device)
        assert a.dtype == data.dtype
        # block_inds
        assert data.block_inds.shape == (len(data.blocks), 2)
        assert np.all(data.block_inds >= 0)
        assert np.all(data.block_inds[:, 0] < a.codomain.num_sectors)
        assert np.all(data.block_inds[:, 1] < a.domain.num_sectors)
        assert np.all(np.lexsort(data.block_inds.T) == np.arange(len(data.blocks)))
        # check charge rule (matching coupled sectors)
        coupled_codomain = a.codomain.sector_decomposition[data.block_inds[:, 0]]
        coupled_domain = a.domain.sector_decomposition[data.block_inds[:, 1]]
        assert np.all(coupled_codomain == coupled_domain)
        # blocks
        for (i, j), block in zip(data.block_inds, data.blocks):
            assert 0 <= i < a.codomain.num_sectors
            assert 0 <= j < a.domain.num_sectors
            expect_shape = (a.codomain.multiplicities[i], a.domain.multiplicities[j])
            if is_diagonal:
                assert expect_shape[0] == expect_shape[1]
                expect_shape = (expect_shape[0],)
            assert all(dim > 0 for dim in expect_shape), 'should skip forbidden block'
            self.block_backend.test_block_sanity(block, expect_shape=expect_shape,
                                                 expect_dtype=a.dtype, expect_device=a.device)

    def test_mask_sanity(self, a: Mask):
        super().test_mask_sanity(a)
        data: FusionTreeData = a.data
        # check device and dtype
        assert a.device == a.data.device == self.block_backend.as_device(a.data.device)
        assert a.dtype == data.dtype == Dtype.bool
        # block_inds
        assert data.block_inds.shape == (len(data.blocks), 2)
        assert np.all(data.block_inds >= 0)
        assert np.all(data.block_inds[:, 0] < a.codomain.num_sectors)
        assert np.all(data.block_inds[:, 1] < a.domain.num_sectors)
        assert np.all(np.lexsort(data.block_inds.T) == np.arange(len(data.blocks)))
        # check charge rule (matching coupled sectors)
        coupled_codomain = a.codomain.sector_decomposition[data.block_inds[:, 0]]
        coupled_domain = a.domain.sector_decomposition[data.block_inds[:, 1]]
        assert np.all(coupled_codomain == coupled_domain)
        # blocks
        for (i, j), block in zip(data.block_inds, data.blocks):
            if a.is_projection:
                expect_len = a.domain.multiplicities[j]
                expect_sum = a.codomain.multiplicities[i]
            else:
                expect_len = a.codomain.multiplicities[i]
                expect_sum = a.domain.multiplicities[j]
            assert expect_len > 0
            assert expect_sum > 0
            self.block_backend.test_block_sanity(block, expect_shape=(expect_len,),
                                                 expect_dtype=Dtype.bool, expect_device=a.device)
            assert self.block_backend.sum_all(block) == expect_sum

    # ABSTRACT METHODS

    def act_block_diagonal_square_matrix(self, a: SymmetricTensor,
                                         block_method: Callable[[Block], Block],
                                         dtype_map: Callable[[Dtype], Dtype] | None) -> Data:
        block_inds = a.data.block_inds
        res_blocks = []
        # square matrix => codomain == domain have the same sectors
        n = 0
        bi = -1 if n >= len(block_inds) else block_inds[n, 0]
        for i in range(a.codomain.num_sectors):
            if bi == i:
                block = a.data.blocks[n]
                n += 1
                bi = -1 if n >= len(block_inds) else block_inds[n, 0]
            else:
                mult = a.codomain.multiplicities[i]
                block = self.block_backend.zeros(shape=[mult, mult], dtype=a.dtype)
            res_blocks.append(block_method(block))
        if dtype_map is None:
            dtype = a.dtype
        else:
            dtype = dtype_map(a.dtype)
        res_block_inds = np.repeat(np.arange(a.domain.num_sectors)[:, None], 2, axis=1)
        return FusionTreeData(res_block_inds, res_blocks, dtype, a.data.device)

    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: TensorProduct, new_domain: TensorProduct
                        ) -> Data:
        # does not change blocks or coupled sectors at all.
        return a.data

    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float
                     ) -> bool:
        # since the coupled sector must agree, it is enough to compare block_inds[:, 0]
        for i, j in iter_common_noncommon_sorted(a.data.block_inds[:, 0], b.data.block_inds[:, 0]):
            if j is None:
                if self.block_backend.max_abs(a.data.blocks[i]) > atol:
                    return False
            elif i is None:
                if self.block_backend.max_abs(b.data.blocks[j]) > atol:
                    return False
            else:
                if not self.block_backend.allclose(a.data.blocks[i], b.data.blocks[j], rtol=rtol, atol=atol):
                    return False
        return True

    def apply_instructions(self, tensor: SymmetricTensor, instructions: Iterable[Instruction],
                           codomain_idcs: list[int], domain_idcs: list[int],
                           new_codomain: TensorProduct, new_domain: TensorProduct,
                           mixes_codomain_domain: bool) -> FusionTreeData:
        """Apply a sequence of :class:`Instruction` s to a tensor.

        Parameters
        ----------
        tensor : SymmetricTensor
            The tensor to act on.
        instruction : list of :class:`Instruction`
            A list of instructions to apply.
        codomain_idcs, domain_idcs : list of int
            The permutation of legs induced by the instructions.
            ``(co)domain_idcs[i] == j`` means that the leg ``tensor.legs[j]`` ends up at
            ``result.(co)domain[i]``
        new_codomain, new_domain : :class:`TensorProduct`
            The (co)domain of the result.
        mixes_codomain_domain : bool
            If any leg moves from codomain to domain or vv during the permutation.

        Returns
        -------
        The :class:`FusionTreeData` for the result of applying the `instructions` to `tensor`.
        """
        cls = TreePairMapping if mixes_codomain_domain else FactorizedTreeMapping
        mapping = cls.from_instructions(instructions=instructions, codomain=tensor.codomain,
                                        domain=tensor.domain, block_inds=tensor.data.block_inds)
        data = mapping.transform_tensor(
            tensor.data, codomain=tensor.codomain, domain=tensor.domain, new_codomain=new_codomain,
            new_domain=new_domain, codomain_idcs=codomain_idcs, domain_idcs=domain_idcs,
            block_backend=self.block_backend
        )
        return data

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        tensor_blocks = tensor.data.blocks
        tensor_block_inds_contr = tensor.data.block_inds[:, :1]  # is sorted
        mask_blocks = mask.data.blocks
        mask_block_inds = mask.data.block_inds
        mask_block_inds_contr = mask_block_inds[:, 1]  # is sorted
        res_blocks = []
        res_block_inds = []  # append only for one leg, repeat later
        for i, j in iter_common_sorted(tensor_block_inds_contr, mask_block_inds_contr):
            block = self.block_backend.apply_mask(tensor_blocks[i], mask_blocks[j], ax=0)
            res_blocks.append(block)
            res_block_inds.append(mask_block_inds[j, 0])
        if len(res_block_inds) > 0:
            res_block_inds = np.repeat(np.array(res_block_inds)[:, None], 2, axis=1)
        else:
            res_block_inds = np.zeros((0, 2), int)
        return FusionTreeData(res_block_inds, res_blocks, tensor.dtype, tensor.data.device,
                              is_sorted=True)

    def combine_legs(self,
                     tensor: SymmetricTensor,
                     leg_idcs_combine: list[list[int]],
                     pipes: list[LegPipe],
                     new_codomain: TensorProduct,
                     new_domain: TensorProduct,
                     ) -> Data:
        raise NotImplementedError('FusionTreeBackend.combine_legs not implemented')

    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        res_dtype = Dtype.common(a.dtype, b.dtype)
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        b_blocks = b.data.blocks
        b_block_inds = b.data.block_inds
        if a.dtype != res_dtype:
            a_blocks = [self.block_backend.to_dtype(bl, res_dtype) for bl in a_blocks]
        if b.dtype != res_dtype:
            b_blocks = [self.block_backend.to_dtype(bl, res_dtype) for bl in b_blocks]
        blocks = []
        block_inds = []
        if len(a.data.block_inds) > 0 and len(b.data.block_inds) > 0:
            for i, j in iter_common_sorted(a.data.block_inds[:, 1], b.data.block_inds[:, 0]):
                blocks.append(self.block_backend.matrix_dot(a_blocks[i], b_blocks[j]))
                block_inds.append([a_block_inds[i, 0], b_block_inds[j, 1]])
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, res_dtype, a.data.device)

    def copy_data(self, a: SymmetricTensor, device: str = None) -> FusionTreeData:
        blocks = [self.block_backend.copy_block(block, device=device) for block in a.data.blocks]
        if device is None:
            device = a.data.device
        else:
            device = self.block_backend.as_device(device)
        return FusionTreeData(
            block_inds=a.data.block_inds.copy(),  # OPTIMIZE do we need to copy these?
            blocks=blocks, dtype=a.data.dtype, device=device
        )

    def dagger(self, a: SymmetricTensor) -> Data:
        return FusionTreeData(
            block_inds=a.data.block_inds[:, ::-1],  # domain and codomain have swapped
            blocks=[self.block_backend.dagger(b) for b in a.data.blocks],
            dtype=a.dtype, device=a.data.device
        )

    def data_item(self, a: FusionTreeData) -> float | complex:
        if len(a.blocks) > 1:
            raise ValueError("More than 1 block!")
        if len(a.blocks) == 0:
            return a.dtype.zero_scalar
        return self.block_backend.item(a.blocks[0])

    def diagonal_all(self, a: DiagonalTensor) -> bool:
        if len(a.data.blocks) < a.domain.num_sectors:
            # there are missing blocks. -> they contain False -> all(a) == False
            return False
        # now it is enough to check the existing blocks
        return all(self.block_backend.block_all(b) for b in a.data.blocks)

    def diagonal_any(self, a: DiagonalTensor) -> bool:
        return any(self.block_backend.block_any(b) for b in a.data.blocks)

    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool) -> DiagonalData:
        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds
        if partial_zero_is_zero:
            blocks = []
            block_inds = []
            for i, j in iter_common_sorted(a_block_inds[:, 0], b_block_inds[:, 0]):
                block_inds.append(a_block_inds[i])
                blocks.append(func(a_blocks[i], b_blocks[j], **func_kwargs))
            if len(block_inds) == 0:
                block_inds = np.zeros((0, 2), int)
            else:
                block_inds = np.array(block_inds, int)
        else:
            n_a = 0  # a_block_inds[:n_a] already visited
            bi_a = -1 if n_a >= len(a_block_inds) else a_block_inds[n_a, 0]
            n_b = 0  # b_block_inds[:n_b] already visited
            bi_b = -1 if n_b >= len(b_block_inds) else b_block_inds[n_b, 0]
            blocks = []
            for i in range(a.codomain.num_sectors):
                if i == bi_a:
                    a_block = a_blocks[n_a]
                    n_a += 1
                    bi_a = -1 if n_a >= len(a_block_inds) else a_block_inds[n_a, 0]
                else:
                    a_block = self.block_backend.zeros([a.domain.multiplicities[i]], dtype=a.dtype)
                if i == bi_b:
                    b_block = b_blocks[n_b]
                    n_b += 1
                    bi_b = -1 if n_b >= len(b_block_inds) else b_block_inds[n_b, 0]
                else:
                    b_block = self.block_backend.zeros([a.domain.multiplicities[i]], dtype=b.dtype)
                blocks.append(func(a_block, b_block, **func_kwargs))
            block_inds = np.repeat(np.arange(a.domain.num_sectors)[:, None], 2, axis=1)
        if len(blocks) > 0:
            dtype = self.block_backend.get_dtype(blocks[0])
        else:
            a_block = self.block_backend.ones_block([1], dtype=a.dtype)
            b_block = self.block_backend.ones_block([1], dtype=b.dtype)
            example_block = func(a_block, b_block, **func_kwargs)
            dtype = self.block_backend.get_dtype(example_block)
        return FusionTreeData(block_inds=block_inds, blocks=blocks, dtype=dtype,
                              device=a.data.device)

    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs,
                                   maps_zero_to_zero: bool) -> DiagonalData:
        if maps_zero_to_zero:
            blocks = [func(b, **func_kwargs) for b in a.data.blocks]
            block_inds = a.data.block_inds
        else:
            a_blocks = a.data.blocks
            a_block_inds = a.data.block_inds
            n = 0
            bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            blocks = []
            for i in range(a.codomain.num_sectors):
                if i == bi:
                    block = a_blocks[n]
                    n += 1
                    bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
                else:
                    mult = a.codomain.multiplicities[i]
                    block = self.block_backend.zeros([mult], dtype=a.dtype)
                blocks.append(func(block, **func_kwargs))
            block_inds = np.repeat(np.arange(a.codomain.num_sectors)[:, None], 2, axis=1)
        if len(blocks) > 0:
            dtype = self.block_backend.get_dtype(blocks[0])
        else:
            example_block = func(self.block_backend.ones_block([1], dtype=a.dtype), **func_kwargs)
            dtype = self.block_backend.get_dtype(example_block)
        return FusionTreeData(block_inds=block_inds, blocks=blocks, dtype=dtype,
                              device=a.data.device)

    def diagonal_from_block(self, a: Block, co_domain: TensorProduct, tol: float) -> DiagonalData:
        leg = co_domain[0]
        dtype = self.block_backend.get_dtype(a)
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        blocks = []
        for coupled, mult in zip(co_domain.sector_decomposition, co_domain.multiplicities):
            dim_c = co_domain.symmetry.sector_dim(coupled)
            # OPTIMIZE (JU) this lookup is annoying, but currently needed because of potentially
            #               different sorting order of the ``a.domain == TensorProduct(a.leg)``
            #               versus the ``a.leg`` itself
            j = leg.sector_decomposition_where(coupled)
            entries = self.block_backend.reshape(a[slice(*leg.slices[j])], (dim_c, mult))
            # project onto the identity on the coupled sector
            block = self.block_backend.sum(entries, 0) / dim_c
            projected = self.block_backend.outer(
                self.block_backend.ones_block([dim_c], dtype=dtype), block
            )
            if self.block_backend.norm(entries - projected) > tol * self.block_backend.norm(entries):
                raise ValueError('Block is not symmetric up to tolerance.')
            blocks.append(block)
        return FusionTreeData(block_inds, blocks, dtype,
                              device=self.block_backend.get_device(a))

    def diagonal_from_sector_block_func(self, func, co_domain: TensorProduct) -> DiagonalData:
        blocks = [func((co_domain.block_size(coupled_idx),), coupled)
                  for coupled_idx, coupled in enumerate(co_domain.sector_decomposition)]
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        if len(blocks) > 0:
            sample_block = blocks[0]
        else:
            sample_block = func((1,), co_domain.symmetry.trivial_sector)
        dtype = self.block_backend.get_dtype(sample_block)
        device = self.block_backend.get_device(sample_block)
        return FusionTreeData(block_inds, blocks, dtype, device)

    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool
                                         ) -> DiagonalData:
        blocks = [self.block_backend.get_diagonal(block, check_offdiagonal)
                  for block in a.data.blocks]
        return FusionTreeData(a.data.block_inds, blocks, a.dtype, a.data.device, is_sorted=True)

    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        return sum(
            (a.domain.sector_qdims[bi] * self.block_backend.sum_all(block)
             for bi, block in zip(a.data.block_inds[:, 0], a.data.blocks)),
            a.dtype.zero_scalar
        )

    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        assert a.symmetry.can_be_dropped
        res = self.block_backend.zeros([a.leg.dim], a.dtype)
        for n, bi in enumerate(a.data.block_inds[:, 0]):
            c = a.codomain.sector_decomposition[bi]
            dim_c = a.codomain.sector_dims[bi]
            symmetry_data = self.block_backend.ones_block([dim_c], dtype=a.dtype)
            degeneracy_data = a.data.blocks[n]
            entries = self.block_backend.outer(symmetry_data, degeneracy_data)
            entries = self.block_backend.reshape(entries, (-1,))
            # OPTIMIZE (JU) this lookup is annoying, but currently needed because of potentially
            #               different sorting order of the ``a.domain == TensorProduct(a.leg)``
            #               versus the ``a.leg`` itself
            j = a.leg.sector_decomposition_where(c)
            res[slice(*a.leg.slices[j])] = entries
        return res

    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[DiagonalData, ElementarySpace]:
        large_leg = tens.leg
        basis_perm = large_leg._basis_perm
        blocks = []
        codom_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        # block_inds are w.r.t. TensorProducts, not the legs
        # -> maybe need to do additional sorting if leg is dual
        is_sorted = not large_leg.is_dual
        for diag_block, diag_bi in zip(tens.data.blocks, tens.data.block_inds):
            if not self.block_backend.block_any(diag_block):
                continue
            bi, _ = diag_bi

            # get the defining sector
            sector = tens.codomain.sector_decomposition[bi]
            if large_leg.is_dual:
                sector = large_leg.symmetry.dual_sector(sector)

            blocks.append(diag_block)
            codom_block_inds.append(bi)
            sectors.append(sector)
            multiplicities.append(self.block_backend.sum_all(diag_block))
            if basis_perm is not None:
                dim = large_leg.symmetry.sector_dim(sector)
                mask = np.tile(self.block_backend.to_numpy(diag_block, bool), dim)
                if large_leg.is_dual:
                    bi = large_leg.sector_decomposition_where(tens.codomain.sector_decomposition[bi])
                basis_perm_ranks.append(basis_perm[slice(*large_leg.slices[bi])][mask])

        if len(blocks) == 0:
            sectors = tens.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros((0, 2), int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if not is_sorted:
                perm = np.lexsort(sectors.T)
                sectors = sectors[perm]
                multiplicities = multiplicities[perm]
            if basis_perm is not None:
                if not is_sorted:
                    basis_perm_ranks = [basis_perm_ranks[p] for p in perm]
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            block_inds = np.column_stack([np.arange(len(sectors)), codom_block_inds])

        data = FusionTreeData(
            block_inds=block_inds, blocks=blocks, dtype=Dtype.bool, device=tens.data.device, is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=tens.symmetry, defining_sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg

    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        # result has block associated with coupled sector c that is given by the block of tens
        # with coupled sector dual(c).
        # since the TensorProduct.sector_decomposition is always sorted, those corresponding
        # sectors do not appear in the same order.
        
        # OPTIMIZE doing this sorting is duplicate work between here and forming tens.leg.dual
        perm = np.lexsort(tens.symmetry.dual_sectors(tens.domain.sector_decomposition).T)
        data = FusionTreeData(block_inds=inverse_permutation(perm)[tens.data.block_inds],
                              blocks=tens.data.blocks, dtype=tens.dtype, device=tens.data.device)
        return tens.leg.dual, data

    def eigh(self, a: SymmetricTensor, new_leg_dual: bool, sort: str = None
             ) -> tuple[DiagonalData, Data, ElementarySpace]:
        new_leg = a.domain.as_ElementarySpace(is_dual=new_leg_dual)
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        v_blocks = []
        w_blocks = []
        n = 0
        bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        for i in range(a.codomain.num_sectors):
            if i == bi:
                vals, vects = self.block_backend.eigh(a_blocks[n], sort=sort)
                v_blocks.append(vects)
                w_blocks.append(vals)
                n += 1
                bi = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                # there is not block for that sector. => eigenvalues are 0.
                # choose eigenvectors as standard basis vectors (eye matrix)
                block_size = a.codomain.multiplicities[i]
                v_blocks.append(self.block_backend.eye_matrix(block_size, a.dtype))
        #
        v_block_inds = np.repeat(np.arange(a.codomain.num_sectors)[:, None], 2, axis=1)
        v_data = FusionTreeData(v_block_inds, v_blocks, a.dtype, a.data.device)
        w_data = FusionTreeData(a_block_inds, w_blocks, a.dtype.to_real, a.data.device)
        return w_data, v_data, new_leg

    def eye_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> FusionTreeData:
        # Note: the identity has the same matrix elements in all ONB, so no need to consider
        #       the basis perms.
        blocks = [self.block_backend.eye_matrix(co_domain.block_size(c_idx), dtype, device)
                  for c_idx in range(co_domain.num_sectors)]
        block_inds = np.repeat(np.arange(co_domain.num_sectors)[:, None], 2, axis=1)
        return FusionTreeData(block_inds, blocks, dtype, device)

    def from_dense_block(self, a: Block, codomain: TensorProduct, domain: TensorProduct, tol: float
                         ) -> FusionTreeData:
        sym = codomain.symmetry
        assert sym.can_be_dropped
        # convert to internal basis order, where the sectors are sorted and contiguous
        J = codomain.num_factors
        K = domain.num_factors
        num_legs = J + K
        # [i1,...,iJ,jK,...,j1] -> [i1,...,iJ,j1,...,jK]
        a = self.block_backend.permute_axes(a, [*range(J), *reversed(range(J, num_legs))])
        dtype = Dtype.common(self.block_backend.get_dtype(a), sym.fusion_tensor_dtype)
        # main loop: iterate over coupled sectors and construct the respective block.
        block_inds = []
        blocks = []
        norm_sq_projected = 0
        for i, j in iter_common_sorted_arrays(codomain.sector_decomposition, domain.sector_decomposition):
            coupled = codomain.sector_decomposition[i]
            dim_c = codomain.sector_dims[i]
            block_size = [codomain.multiplicities[i], domain.multiplicities[j]]
            # OPTIMIZE could be sth like np.empty
            block = self.block_backend.zeros(block_size, dtype)
            # iterate over uncoupled sectors / forest-blocks within the block
            i1 = 0  # start row index of the current forest block
            i2 = 0  # start column index of the current forest block
            for b_sectors, n_dims, j2 in domain.iter_uncoupled(yield_slices=True):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = domain.tree_block_size(b_sectors)
                for a_sectors, m_dims, j1 in codomain.iter_uncoupled(yield_slices=True):
                    a_dims = sym.batch_sector_dim(a_sectors)
                    tree_block_height = codomain.tree_block_size(a_sectors)
                    entries = a[(*j1, *j2)]  # [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
                    # reshape to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
                    shape = [0] * (2 * num_legs)
                    shape[::2] = [*a_dims, *b_dims]
                    shape[1::2] = [*m_dims, *n_dims]
                    entries = self.block_backend.reshape(entries, shape)
                    # permute to [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...nK]
                    perm = [*range(0, 2 * num_legs, 2), *range(1, 2 * num_legs, 2)]
                    entries = self.block_backend.permute_axes(entries, perm)
                    num_alpha_trees, num_beta_trees = self._add_forest_block_entries(
                        block, entries, sym, codomain, domain, coupled, dim_c, a_sectors, b_sectors,
                        tree_block_width, tree_block_height, i1, i2
                    )
                    forest_block_height = num_alpha_trees * tree_block_height
                    forest_block_width = num_beta_trees * tree_block_width
                    i1 += forest_block_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_block_width  # move right by one forest-block
            block_norm = self.block_backend.norm(block, order=2)
            if block_norm <= 0.:  # TODO small finite tolerance instead?
                continue
            block_inds.append([i, j])
            blocks.append(block)
            contribution = dim_c * block_norm ** 2
            norm_sq_projected += contribution

        # since the symmetric and non-symmetric components of ``a = a_sym + a_rest`` are mutually
        # orthogonal, we have  ``norm(a) ** 2 = norm(a_sym) ** 2 + norm(a_rest) ** 2``.
        # thus ``abs_err = norm(a - a_sym) = norm(a_rest) = sqrt(norm(a) ** 2 - norm(a_sym) ** 2)``
        if tol is not None:
            a_norm_sq = self.block_backend.norm(a, order=2) ** 2
            norm_diff_sq = a_norm_sq - norm_sq_projected
            abs_tol_sq = tol * tol * a_norm_sq
            if norm_diff_sq > abs_tol_sq > 0:
                msg = (f'Block is not symmetric up to tolerance. '
                       f'Original norm: {np.sqrt(a_norm_sq)}. '
                       f'Norm after projection: {np.sqrt(norm_sq_projected)}.')
                raise ValueError(msg)
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, dtype,
                              device=self.block_backend.get_device(block))

    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        raise NotImplementedError('from_dense_block_trivial_sector not implemented')

    def from_grid(self, grid: list[list[SymmetricTensor | None]], new_codomain: TensorProduct,
                  new_domain: TensorProduct, left_mult_slices: list[list[int]],
                  right_mult_slices: list[list[int]], dtype: Dtype, device: str) -> Data:
        # idea: iterate over forest blocks of the operators in the grid; forest blocks are always
        # contiguous except in the case where contributions to the same forest block comes from
        # different operators -> reshape such that the legs along which we stack are isolated
        data = self.zero_data(new_codomain, new_domain, dtype=dtype, device=device, all_blocks=True)
        for i, row in enumerate(grid):
            for j, op in enumerate(row):
                if op is None:
                    continue
                op_coupled = [op.domain.sector_decomposition[bi[1]] for bi in op.data.block_inds]
                for codom_unc, op_codom_slc, op_coupled_idx in op.codomain.iter_forest_blocks(op_coupled):
                    coupled = op_coupled[op_coupled_idx]
                    block_idx = data.block_ind_from_coupled(coupled, new_domain)

                    # goal: reshape co_domain part such that it has 3 axes:
                    # for the trees, for the multiplicity of codomain[0] / domain[-1], for all other multiplicities
                    op_codom_shape = [op.codomain[l].sector_multiplicity(sec) for l, sec in enumerate(codom_unc)]
                    op_codom_shape = [op_codom_shape[0], prod(op_codom_shape[1:])]
                    op_num_codom_trees = int((op_codom_slc.stop - op_codom_slc.start) / prod(op_codom_shape))

                    # only the first space is different
                    codom_shape = [new_codomain[0].sector_multiplicity(codom_unc[0])] + op_codom_shape[1:]
                    codom_slc = new_codomain.forest_block_slice(codom_unc, coupled)
                    num_codom_trees = int((codom_slc.stop - codom_slc.start) / prod(codom_shape))
                    codom_leg_idx = new_codomain[0].sector_decomposition_where(codom_unc[0])
                    codom_leg_slc = slice(left_mult_slices[codom_leg_idx][i], left_mult_slices[codom_leg_idx][i + 1])
                    for dom_unc, op_dom_slc, _ in op.domain.iter_forest_blocks([coupled]):
                        op_dom_shape = [op.domain[l].sector_multiplicity(sec) for l, sec in enumerate(dom_unc)]
                        op_dom_shape = [prod(op_dom_shape[:-1]), op_dom_shape[-1]]
                        op_num_dom_trees = int((op_dom_slc.stop - op_dom_slc.start) / prod(op_dom_shape))
                        op_new_shape = (op_num_codom_trees, *op_codom_shape, op_num_dom_trees, *op_dom_shape)

                        # only the last space is different
                        dom_shape = op_dom_shape[:1] + [new_domain[-1].sector_multiplicity(dom_unc[-1])]
                        dom_slc = new_domain.forest_block_slice(dom_unc, coupled)
                        num_dom_trees = int((dom_slc.stop - dom_slc.start) / prod(dom_shape))
                        new_shape = (num_codom_trees, *codom_shape, num_dom_trees, *dom_shape)
                        dom_leg_idx = new_domain[-1].sector_decomposition_where(dom_unc[-1])
                        dom_leg_slc = slice(right_mult_slices[dom_leg_idx][j], right_mult_slices[dom_leg_idx][j + 1])

                        op_block = op.data.blocks[op_coupled_idx][op_codom_slc, op_dom_slc]
                        op_block = self.block_backend.reshape(op_block, op_new_shape)

                        block = self.block_backend.copy_block(data.blocks[block_idx][codom_slc, dom_slc],
                                                              device=device)
                        final_shape = self.block_backend.get_shape(block)
                        block = self.block_backend.reshape(block, new_shape)
                        block[:, codom_leg_slc, :, :, :, dom_leg_slc] += op_block
                        block = self.block_backend.reshape(block, final_shape)
                        data.blocks[block_idx][codom_slc, dom_slc] = block
        data.discard_zero_blocks(self.block_backend, self.eps)
        return data

    def from_random_normal(self, codomain: TensorProduct, domain: TensorProduct, sigma: float,
                           dtype: Dtype, device: str) -> Data:
        def func(shape, coupled):
            return self.block_backend.random_normal(shape, dtype, sigma, device=device)
        
        return self.from_sector_block_func(func, codomain=codomain, domain=domain)

    def from_sector_block_func(self, func, codomain: TensorProduct, domain: TensorProduct) -> FusionTreeData:
        blocks = []
        block_inds = []
        for i, j in iter_common_sorted_arrays(codomain.sector_decomposition, domain.sector_decomposition):
            coupled = codomain.sector_decomposition[i]
            shape = (codomain.block_size(i), domain.block_size(j))
            block_inds.append([i, j])
            blocks.append(func(shape, coupled))
        if len(blocks) > 0:
            sample_block = blocks[0]
            block_inds = np.asarray(block_inds, int)
        else:
            sample_block = func((1, 1), codomain.symmetry.trivial_sector)
            block_inds = np.zeros((0, 2), int)
        dtype = self.block_backend.get_dtype(sample_block)
        device = self.block_backend.get_device(sample_block)
        return FusionTreeData(block_inds, blocks, dtype, device)

    def from_tree_pairs(self, trees: dict[tuple[FusionTree, FusionTree], Block], codomain: TensorProduct,
                        domain: TensorProduct, dtype: Dtype, device: str) -> Data:
        J = codomain.num_factors
        K = domain.num_factors
        block_inds = []
        blocks = []
        pairs_done = set()
        for i, j in iter_common_sorted_arrays(codomain.sector_decomposition, domain.sector_decomposition):
            coupled = codomain.sector_decomposition[i]
            shape = codomain.multiplicities[i], domain.multiplicities[j]
            block = self.block_backend.zeros(shape, dtype, device)
            is_zero_block = True
            for Y, i1, mults1, _ in codomain.iter_tree_blocks([coupled]):
                for X, i2, mults2, _ in domain.iter_tree_blocks([coupled]):
                    pair = (Y, X)
                    tree_block = trees.get(pair, None)
                    if tree_block is None:
                        continue
                    expect_shape = *mults1, *reversed(mults2)
                    assert self.block_backend.get_shape(tree_block) == expect_shape
                    # [m1,...,mJ,nK,...,n1] -> [m1,...,mJ,n1,...,nK]
                    tree_block = self.block_backend.permute_axes(tree_block, [*range(J), *reversed(range(J, J + K))])
                    # [m1,...,mJ,n1,...,nK] -> [M, N]
                    tree_block = self.block_backend.reshape(tree_block, (np.prod(mults1), np.prod(mults2)))
                    block[i1, i2] = tree_block
                    is_zero_block = False
                    pairs_done.add(pair)
            if is_zero_block:
                continue
            block_inds.append([i, j])
            blocks.append(block)
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds)
        # check if we covered all keys in the dict
        for pair in trees.keys():
            if pair not in pairs_done:
                # OPTIMIZE if the code works, we could remove this check
                raise RuntimeError
        return FusionTreeData(block_inds=block_inds, blocks=blocks, dtype=dtype, device=device,
                              is_sorted=True)

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_backend.block_from_diagonal(block) for block in a.data.blocks]
        return FusionTreeData(a.data.block_inds, blocks, dtype=a.dtype, device=a.data.device)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        blocks = [self.block_backend.block_from_mask(block, dtype) for block in a.data.blocks]
        return FusionTreeData(block_inds=a.data.block_inds, blocks=blocks,
                              dtype=dtype, device=a.data.device, is_sorted=True)

    def get_device_from_data(self, a: FusionTreeData) -> str:
        return a.device

    def get_dtype_from_data(self, a: FusionTreeData) -> Dtype:
        return a.dtype

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        msg = ('Accessing individual entries in the FusionTreeBackend is comparably '
               'expensive. When accessing multiple entries, it may be more efficient '
               'to use to_numpy() first and then access the entries of the tensor.')
        warnings.warn(msg, UserWarning, stacklevel=2)

        num_cod_legs = a.num_codomain_legs
        num_legs = a.num_legs
        # reverse domain idcs -> work in the non-conventional leg order [i1,...,iJ,j1,...,jK]
        a_legs = [*a.codomain.factors, *a.domain.factors]
        idcs = idcs[:num_cod_legs] + idcs[num_cod_legs:][::-1]
        pos = np.array([l.parse_index(idx) for l, idx in zip(a_legs, idcs)])
        sector_idcs = pos[:, 0]

        uncoupled = np.array([l.sector_decomposition[sector_idcs[i]] for i, l in enumerate(a_legs)])
        codom_uncoupled = uncoupled[:num_cod_legs, :]
        dom_uncoupled = uncoupled[num_cod_legs:, :]
        mults = [l.multiplicities[sector_idcs[i]] for i, l in enumerate(a_legs)]
        codom_mults = mults[:num_cod_legs]
        dom_mults = mults[num_cod_legs:]
        dims = a.symmetry.batch_sector_dim(uncoupled)
        codom_dims = dims[:num_cod_legs]
        dom_dims = dims[num_cod_legs:]

        # build the correct forest block to get the element
        shape = [d * m for d, m in zip(dims, mults)]
        dtype = Dtype.common(a.data.dtype, a.symmetry.fusion_tensor_dtype)
        forest_block = self.block_backend.zeros(shape, dtype=dtype, device=a.device)
        tree_block_height = a.codomain.tree_block_size(codom_uncoupled)
        tree_block_width = a.domain.tree_block_size(dom_uncoupled)
        for bi_cod, block in zip(a.data.block_inds[:, 0], a.data.blocks):
            coupled = a.codomain.sector_decomposition[bi_cod]
            i1 = a.codomain.forest_block_slice(codom_uncoupled, coupled).start
            i2 = a.domain.forest_block_slice(dom_uncoupled, coupled).start
            entries, _, _ = self._get_forest_block_contribution(
                block, a.symmetry, a.codomain, a.domain, coupled, codom_uncoupled, dom_uncoupled,
                codom_dims, dom_dims, tree_block_width, tree_block_height, i1, i2, codom_mults,
                dom_mults, dtype
            )
            # entries : [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
            # permute to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
            perm = [i + offset for i in range(num_legs) for offset in [0, num_legs]]
            entries = self.block_backend.permute_axes(entries, perm)
            # reshape to [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
            entries = self.block_backend.reshape(entries, shape)
            forest_block += entries
        return self.block_backend.get_block_element(forest_block, pos[:, 1])

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        sector_idx, idx_within = a.leg.parse_index(idx)
        multi = a.leg.multiplicities[sector_idx]
        if a.leg.is_dual:
            sector = a.leg.sector_decomposition[sector_idx]
            sector_idx = a.domain.sector_decomposition_where(sector)
        block_idx = a.data.block_ind_from_domain_sector_ind(sector_idx)
        if block_idx is None:
            return a.dtype.zero_scalar
        block = a.data.blocks[block_idx]
        return self.block_backend.get_block_element(block, [idx_within % multi])

    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        pos = np.array([l.parse_index(idx) for l, idx in zip(conventional_leg_order(a), idcs)])
        sector_idx = pos[1, 0]  # domain leg index
        sector = a.domain[0].sector_decomposition[sector_idx]
        if not all(sector == a.codomain[0].sector_decomposition[pos[0, 0]]):
            return False
        if a.domain[0].is_dual:
            sector_idx = a.domain.sector_decomposition_where(sector)
        block_idx = a.data.block_ind_from_domain_sector_ind(sector_idx)
        if block_idx is None:
            return False
        block = a.data.blocks[block_idx]
        if a.is_projection:
            small, large = pos[:, 1]
            multi = a.small_leg.multiplicities[pos[0, 0]]
        else:
            large, small = pos[:, 1]
            multi = a.small_leg.multiplicities[pos[1, 0]]
        return self.block_backend.get_block_mask_element(block, large, small, sum_block=multi)

    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        a_blocks = a.data.blocks
        a_codomain_qdims = a.codomain.sector_qdims
        b_blocks = b.data.blocks
        a_codomain_block_inds = a.data.block_inds[:, 0]
        if do_dagger:
            # need to match a.codomain == b.codomain
            b_block_inds = b.data.block_inds[:, 0]
        else:
            # need to math a.codomain == b.domain
            b_block_inds = b.data.block_inds[:, 1]
        res = a.dtype.zero_scalar * b.dtype.zero_scalar
        for i, j in iter_common_sorted(a_codomain_block_inds, b_block_inds):
            inn = self.block_backend.inner(a_blocks[i], b_blocks[j], do_dagger=do_dagger)
            res += a_codomain_qdims[a_codomain_block_inds[i]] * inn
        return res

    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                                charge_leg: ElementarySpace) -> Data:
        raise NotImplementedError('inv_part_from_dense_block_single_sector not implemented')

    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('inv_part_to_dense_block_single_sector not implemented')

    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        dtype = v.data.dtype.common(w.data.dtype)
        v_blocks = [self.block_backend.to_dtype(_a, dtype) for _a in v.data.blocks]
        w_blocks = [self.block_backend.to_dtype(_b, dtype) for _b in w.data.blocks]
        v_block_inds = v.data.block_inds
        w_block_inds = w.data.block_inds
        blocks = []
        block_inds = []
        for i, j in iter_common_noncommon_sorted(v_block_inds[:, 0], w_block_inds[:, 0]):
            if i is None:
                blocks.append(self.block_backend.mul(b, w_blocks[j]))
                block_inds.append(w_block_inds[j])
            elif j is None:
                blocks.append(self.block_backend.mul(a, v_blocks[i]))
                block_inds.append(v_block_inds[i])
            else:
                blocks.append(
                    self.block_backend.linear_combination(a, v_blocks[i], b, w_blocks[j])
                )
                block_inds.append(v_block_inds[i])
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, dtype, device=v.data.device)

    def lq(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        l_blocks = []
        l_block_inds = []
        q_blocks = []
        q_block_inds = []
        n = 0
        bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        itr = iter_common_sorted_arrays(a.codomain.sector_decomposition,
                                        a.domain.sector_decomposition)
        for i_new, (i_cod, i_dom) in enumerate(itr):
            q_block_inds.append([i_new, i_dom])
            if bi_cod == i_cod:
                l, q = self.block_backend.matrix_lq(a_blocks[n], full=False)
                l_blocks.append(l)
                q_blocks.append(q)
                l_block_inds.append([i_cod, i_new])
                n += 1
                bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                B_dom = a.domain.multiplicities[i_dom]
                B_new = new_co_domain.multiplicities[i_new]
                q_blocks.append(self.block_backend.eye_matrix(B_dom, a.dtype)[:B_new, :])
        if len(l_block_inds) == 0:
            l_block_inds = np.zeros((0, 2), int)
        else:
            l_block_inds = np.array(l_block_inds)
        if len(q_block_inds) == 0:
            q_block_inds = np.zeros((0, 2), int)
        else:
            q_block_inds = np.array(q_block_inds)
        l_data = FusionTreeData(l_block_inds, l_blocks, a.dtype, a.data.device)
        q_data = FusionTreeData(q_block_inds, q_blocks, a.dtype, a.data.device)
        return l_data, q_data

    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[MaskData, ElementarySpace]:
        large_leg = mask1.large_leg
        basis_perm = large_leg._basis_perm
        mask1_block_inds = mask1.data.block_inds
        mask1_blocks = mask1.data.blocks
        mask2_block_inds = mask2.data.block_inds
        mask2_blocks = mask2.data.blocks
        #
        blocks = []
        dom_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        # block_inds are w.r.t. TensorProducts, not the legs
        # -> maybe need to do additional sorting and searching if leg is dual
        is_sorted = not large_leg.is_dual
        #
        i1 = 0  # next block of mask1 to process; iterating like this only works if is_sorted.
        b1_i1 = -1 if len(mask1_block_inds) == 0 else mask1_block_inds[i1, 1]  # its block_ind for the large leg.
        i2 = 0
        b2_i2 = -1 if len(mask2_block_inds) == 0 else mask2_block_inds[i2, 1]
        #
        for sector_idx, sector in enumerate(large_leg.defining_sectors):
            if not is_sorted:
                # do this here for both masks
                dual_sec = large_leg.sector_decomposition[sector_idx]
                dom_idx = mask1.domain.sector_decomposition_where(dual_sec)

            block1_found = False
            if is_sorted and sector_idx == b1_i1:
                block1_found = True
                block1 = mask1_blocks[i1]
                i1 += 1
                if i1 >= len(mask1_block_inds):
                    b1_i1 = -1  # mask1 has no further blocks
                else:
                    b1_i1 = mask1_block_inds[i1, 1]
            elif not is_sorted:
                i1 = mask1.data.block_ind_from_domain_sector_ind(dom_idx)
                if i1 is not None:
                    block1_found = True
                    block1 = mask1_blocks[i1]
            if not block1_found:
                block1 = self.block_backend.zeros([large_leg.multiplicities[sector_idx]], Dtype.bool)

            block2_found = False
            if is_sorted and sector_idx == b2_i2:
                block2_found = True
                block2 = mask2_blocks[i2]
                i2 += 1
                if i2 >= len(mask2_block_inds):
                    b2_i2 = -1  # mask2 has no further blocks
                else:
                    b2_i2 = mask2_block_inds[i2, 1]
            elif not is_sorted:
                i2 = mask2.data.block_ind_from_domain_sector_ind(dom_idx)
                if i2 is not None:
                    block2_found = True
                    block2 = mask2_blocks[i2]
            if not block2_found:
                block2 = self.block_backend.zeros([large_leg.multiplicities[sector_idx]], Dtype.bool)

            new_block = func(block1, block2)
            mult = self.block_backend.sum_all(new_block)
            if mult == 0:
                continue
            blocks.append(new_block)
            dom_block_inds.append(sector_idx)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                dim = large_leg.symmetry.sector_dim(sector)
                mask = np.tile(self.block_backend.to_numpy(new_block, bool), dim)
                basis_perm_ranks.append(basis_perm[slice(*large_leg.slices[sector_idx])][mask])

        if len(sectors) == 0:
            sectors = mask1.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros((0, 2), int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            block_inds = np.column_stack([np.arange(len(sectors)), dom_block_inds])
        data = FusionTreeData(
            block_inds=block_inds, blocks=blocks, dtype=Dtype.bool, device=mask1.device, is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=mask1.symmetry, defining_sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg

    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, TensorProduct, TensorProduct]:
        return self._mask_contract(tensor, mask, leg_idx, large_leg=True)

    def mask_contract_small_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, TensorProduct, TensorProduct]:
        return self._mask_contract(tensor, mask, leg_idx, large_leg=False)

    def _mask_contract(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int, large_leg: bool
                       ) -> tuple[Data, TensorProduct, TensorProduct]:
        backend = self.block_backend
        in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)
        in_domain_int = int(in_domain)
        if in_domain:
            assert mask.is_projection != large_leg
        else:
            assert mask.is_projection == large_leg

        if in_domain:
            codomain = tensor.codomain
            spaces = tensor.domain.factors[:]
            spaces[co_domain_idx] = mask.small_leg if large_leg else mask.large_leg
            target_space = domain = TensorProduct(spaces, symmetry=tensor.symmetry)
        else:
            domain = tensor.domain
            spaces = tensor.codomain.factors[:]
            spaces[co_domain_idx] = mask.small_leg if large_leg else mask.large_leg
            target_space = codomain = TensorProduct(spaces, symmetry=tensor.symmetry)

        tensor_blocks = tensor.data.blocks
        tensor_block_inds = tensor.data.block_inds
        mask_blocks = mask.data.blocks

        coupled = [tensor.domain.sector_decomposition[bi[1]] for bi in tensor_block_inds]
        iter_space = tensor.domain if in_domain else tensor.codomain
        same_decomp = len(iter_space.sector_decomposition) == len(target_space.sector_decomposition)
        res_blocks = [backend.zeros([codomain.block_size(c), domain.block_size(c)], tensor.data.dtype)
                      for c in coupled]
        res_block_inds = tensor_block_inds.copy()
        if not same_decomp:
            # sector decomposition changes, need to adjust res_block_inds
            for i, (c, block) in enumerate(zip(coupled, res_blocks)):
                # some coupled sectors may no longer be allowed if uncoupled sectors are projected out
                # -> the corresponding shape has a zero entry -> remove later using discard_zero_blocks
                if backend.get_shape(block)[in_domain_int] > 0:
                    res_block_inds[i, in_domain_int] = target_space.sector_decomposition_where(c)
        
        for uncoupled, slc, i in iter_space.iter_forest_blocks(coupled):
            dom_idx_mask = mask.domain.sector_decomposition_where(uncoupled[co_domain_idx])
            if dom_idx_mask is None:
                continue  # uncoupled sector not in mask
            j = mask.data.block_ind_from_domain_sector_ind(dom_idx_mask)
            if j is None:
                continue  # uncoupled sector not in mask

            intermediate_shape = [iter_space[i].sector_multiplicity(sec)
                                  for i, sec in enumerate(uncoupled)]
            if in_domain:
                block_slice = tensor_blocks[i][:, slc]
                intermediate_shape.insert(0, -1)
                final_shape = (backend.get_shape(block_slice)[0], -1)
            else:
                block_slice = tensor_blocks[i][slc, :]
                intermediate_shape.append(-1)
                final_shape = (-1, backend.get_shape(block_slice)[1])

            block_slice = backend.reshape(block_slice, tuple(intermediate_shape))
            if large_leg:
                block_slice = backend.apply_mask(block_slice, mask_blocks[j],
                                                 ax=in_domain_int + co_domain_idx)
            else:
                block_slice = backend.enlarge_leg(block_slice, mask_blocks[j],
                                                  axis=in_domain_int + co_domain_idx)
            block_slice = backend.reshape(block_slice, final_shape)

            new_slc = target_space.forest_block_slice(uncoupled, coupled[i])
            if in_domain:
                res_blocks[i][:, new_slc] = block_slice
            else:
                res_blocks[i][new_slc, :] = block_slice

        data = FusionTreeData(block_inds=res_block_inds, blocks=res_blocks, dtype=tensor.dtype,
                              device=tensor.device, is_sorted=True)
        data.discard_zero_blocks(self.block_backend, self.eps)
        return data, codomain, domain

    def mask_dagger(self, mask: Mask) -> MaskData:
        # the legs swap between domain and codomain. need to swap the two columns of block_inds.
        # since both columns are unique and ascending, the resulting block_inds are still sorted.
        block_inds = mask.data.block_inds[:, ::-1]
        return FusionTreeData(block_inds=block_inds, blocks=mask.data.blocks, dtype=mask.dtype,
                              device=mask.device, is_sorted=True)

    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        basis_perm = large_leg._basis_perm
        blocks = []
        dom_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        # block_inds are w.r.t. TensorProducts, not the legs
        # -> maybe need to do additional sorting and searching if leg is dual
        is_sorted = not large_leg.is_dual
        if not is_sorted:
            perm = np.lexsort(large_leg.sector_decomposition.T)
            sorted_duals = large_leg.sector_decomposition[perm]
            multis = large_leg.multiplicities[perm]
            domain = TensorProduct([large_leg], symmetry=large_leg.symmetry,
                                   _sector_decomposition=sorted_duals, _multiplicities=multis)
        for bi_large, (slc, sector) in enumerate(zip(large_leg.slices, large_leg.defining_sectors)):
            block = a[slice(*slc)]
            mult = self.block_backend.sum_all(block)
            if mult == 0:
                continue
            if not is_sorted:
                dual_sector = large_leg.symmetry.dual_sector(sector)
                bi_large = domain.sector_decomposition_where(dual_sector)
            dom_block_inds.append(bi_large)
            sectors.append(sector)
            dim = large_leg.symmetry.sector_dim(sector)
            stop = int(len(block) // dim)
            blocks.append(block[:stop])
            multiplicities.append(mult // dim)
            if basis_perm is not None:
                mask = self.block_backend.to_numpy(block)
                basis_perm_ranks.append(large_leg.basis_perm[slice(*slc)][mask])

        if len(sectors) == 0:
            sectors = large_leg.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros((0, 2), int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            if not is_sorted:
                perm = np.argsort(dom_block_inds)
                dom_block_inds = [dom_block_inds[p] for p in perm]
                blocks = [blocks[p] for p in perm]
            block_inds = np.column_stack([np.arange(len(sectors)), dom_block_inds])
        data = FusionTreeData(
            block_inds=block_inds, blocks=blocks, dtype=Dtype.bool,
            device=self.block_backend.get_device(a), is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=large_leg.symmetry, defining_sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg

    def mask_to_block(self, a: Mask) -> Block:
        large_leg = a.large_leg
        res = self.block_backend.zeros([large_leg.dim], Dtype.bool)
        idx = 1 if a.is_projection else 0
        co_dom = a.domain if a.is_projection else a.codomain
        for block, b_i in zip(a.data.blocks, a.data.block_inds):
            bi_large = b_i[idx]
            sector = co_dom.sector_decomposition[bi_large]
            dim = co_dom.symmetry.sector_dim(sector)
            if large_leg.is_dual:
                bi_large = large_leg.sector_decomposition_where(sector)
            res[slice(*large_leg.slices[bi_large])] = a.backend.block_backend.tile(block, dim)
        return res

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        blocks = [self.block_backend.to_dtype(b, dtype) for b in a.data.blocks]
        large_leg_bi = a.data.block_inds[:, 1] if a.is_projection else a.data.block_inds[:, 0]
        block_inds = np.repeat(large_leg_bi[:, None], 2, axis=1)
        return FusionTreeData(block_inds=block_inds, blocks=blocks,
                              dtype=dtype, device=a.data.device, is_sorted=True)

    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        # similar implementation to diagonal_transpose
        # OPTIMIZE doing this sorting is duplicate work between here and forming tens.leg.dual
        block_inds = tens.data.block_inds
        perm_dom = np.lexsort(tens.symmetry.dual_sectors(tens.domain.sector_decomposition).T)
        perm_codom = np.lexsort(tens.symmetry.dual_sectors(tens.codomain.sector_decomposition).T)
        block_inds = np.stack([inverse_permutation(perm_dom)[block_inds[:, 1]],
                               inverse_permutation(perm_codom)[block_inds[:, 0]]], axis=1)
        data = FusionTreeData(block_inds=block_inds, blocks=tens.data.blocks,
                              dtype=tens.dtype, device=tens.data.device, is_sorted=False)
        return tens.codomain[0].dual, tens.domain[0].dual, data

    def mask_unary_operand(self, mask: Mask, func) -> tuple[MaskData, ElementarySpace]:
        large_leg = mask.large_leg
        basis_perm = large_leg._basis_perm
        mask_block_inds = mask.data.block_inds
        mask_blocks = mask.data.blocks
        #
        blocks = []
        dom_block_inds = []
        sectors = []
        multiplicities = []
        basis_perm_ranks = []
        # block_inds are w.r.t. TensorProducts, not the legs
        # -> maybe need to do additional sorting and searching if leg is dual
        is_sorted = not large_leg.is_dual
        #
        i = 0  # next block of mask to process; iterating like this only works if is_sorted.
        b_i = -1 if len(mask_block_inds) == 0 else mask_block_inds[i, 1]
        #
        for sector_idx, sector in enumerate(large_leg.defining_sectors):
            block_found = False
            if is_sorted and sector_idx == b_i:
                block_found = True
                block = mask_blocks[i]
                i += 1
                if i >= len(mask_block_inds):
                    b_i = -1  # mask has no further blocks
                else:
                    b_i = mask_block_inds[i, 1]
            elif not is_sorted:
                dual_sec = large_leg.sector_decomposition[sector_idx]
                i = mask.data.block_ind_from_coupled(dual_sec, mask.domain)
                if i is not None:
                    block_found = True
                    block = mask_blocks[i]
            if not block_found:
                block = self.block_backend.zeros([large_leg.multiplicities[sector_idx]], Dtype.bool)

            new_block = func(block)
            mult = self.block_backend.sum_all(new_block)
            if mult == 0:
                continue
            blocks.append(new_block)
            dom_block_inds.append(sector_idx)
            sectors.append(sector)
            multiplicities.append(mult)
            if basis_perm is not None:
                dim = large_leg.symmetry.sector_dim(sector)
                mask = np.tile(self.block_backend.to_numpy(new_block, bool), dim)
                basis_perm_ranks.append(basis_perm[slice(*large_leg.slices[sector_idx])][mask])

        if len(sectors) == 0:
            sectors = mask.symmetry.empty_sector_array
            multiplicities = np.zeros(0, int)
            basis_perm = None
            block_inds = np.zeros((0, 2), int)
        else:
            sectors = np.array(sectors, int)
            multiplicities = np.array(multiplicities, int)
            if basis_perm is not None:
                basis_perm = rank_data(np.concatenate(basis_perm_ranks))
            block_inds = np.column_stack([np.arange(len(sectors)), dom_block_inds])
        data = FusionTreeData(
            block_inds=block_inds, blocks=blocks, dtype=Dtype.bool, device=mask.device, is_sorted=True
        )
        small_leg = ElementarySpace(
            symmetry=mask.symmetry, defining_sectors=sectors, multiplicities=multiplicities,
            is_dual=large_leg.is_dual, basis_perm=basis_perm
        )
        return data, small_leg

    def move_to_device(self, a: SymmetricTensor | DiagonalTensor | Mask, device: str) -> Data:
        for i in range(len(a.data.blocks)):
            a.data.blocks[i] = self.block_backend.as_block(a.data.blocks[i], device=device)
        a.data.device = self.block_backend.as_device(device)
        return a.data

    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        if a == 0.:
            return self.zero_data(b.codomain, b.domain, b.dtype, device=b.data.device)
        blocks = [self.block_backend.mul(a, T) for T in b.data.blocks]
        if len(blocks) == 0:
            if isinstance(a, float):
                dtype = b.data.dtype
            else:
                dtype = b.data.dtype.to_complex()
        else:
            dtype = self.block_backend.get_dtype(blocks[0])
        return FusionTreeData(b.data.block_inds, blocks, dtype, b.data.device)

    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        # OPTIMIZE should we offer the square-norm instead?
        norm_sq = 0
        for i, block in zip(a.data.block_inds[:, 0], a.data.blocks):
            norm_sq += a.codomain.sector_qdims[i] * (self.block_backend.norm(block) ** 2)
        return np.sqrt(norm_sq).item()

    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        if a.has_pipes or b.has_pipes:
            raise NotImplementedError("'outer' can not deal with 'LegPipe's")
        # idea: get the fusion trees in the combined (co)domain by inserting an identity
        # = summing over all fusion products of the coupled sectors of tensors a and b
        # OPTIMIZE new_codomain and new_domain are already computed in tensors.py -> reuse here
        new_codomain = TensorProduct.from_partial_products(a.codomain, b.codomain)
        new_domain = TensorProduct.from_partial_products(a.domain, b.domain)
        new_data = self.zero_data(new_codomain, new_domain, a.dtype, a.device, all_blocks=True)
        for a_codom_tree, a_dom_tree, a_tree_block in _tree_block_iter(a):
            for b_codom_tree, b_dom_tree, b_tree_block in _tree_block_iter(b):
                # axes of new_tree_block after outer: (a.codomain, a.domain, b.codomain, b.domain)
                new_tree_block = self.block_backend.outer(a_tree_block, b_tree_block)
                new_tree_block = self.block_backend.permute_axes(new_tree_block, [0, 2, 1, 3])
                new_tree_block = self.block_backend.combine_legs(new_tree_block, [[0, 1], [2, 3]])
                #
                new_codom_trees = a_codom_tree.outer(b_codom_tree)
                new_dom_trees = a_dom_tree.outer(b_dom_tree)
                for new_dom_tree, dom_amp in new_dom_trees.items():
                    dom_slc = new_domain.tree_block_slice(new_dom_tree)
                    block_idx = new_data.block_ind_from_coupled(new_dom_tree.coupled, new_domain)
                    for new_codom_tree, codom_amp in new_codom_trees.items():
                        if not all(new_codom_tree.coupled == new_dom_tree.coupled):
                            continue
                        codom_slc = new_codomain.tree_block_slice(new_codom_tree)
                        factor = np.conj(codom_amp) * dom_amp
                        new_data.blocks[block_idx][codom_slc, dom_slc] += new_tree_block * factor
        new_data.discard_zero_blocks(self.block_backend, self.eps)
        return new_data

    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int | None]) -> tuple[Data, TensorProduct, TensorProduct]:
        # step 1: permute legs such that the paired legs are next to each other
        # it does not matter which leg is moved; the partial trace implies that
        # the fusion channel of each pair is trivial. It is however crucial that
        # we keep the ordering within each pair.
        # TODO decide if we want to optimize: There is in principle no need to
        # braid when tracing out two pairs of the form [1, 4] and [2, 3]
        pairs = sorted([pair if pair[0] < pair[1] else (pair[1], pair[0]) for pair in pairs])
        idcs1 = []
        idcs2 = []
        for i1, i2 in pairs:
            idcs1.append(i1)
            idcs2.append(i2)
        remaining = [n for n in range(tensor.num_legs) if n not in idcs1 and n not in idcs2]

        new_codomain = TensorProduct(
            [leg for n, leg in enumerate(tensor.codomain) if n in remaining],
            symmetry=tensor.symmetry
        )
        new_domain = TensorProduct(
            [leg for n, leg in enumerate(tensor.domain) if tensor.num_legs - 1 - n in remaining],
            symmetry=tensor.symmetry
        )

        insert_idcs = [np.searchsorted(remaining, pair[0]) + 2 * i for i, pair in enumerate(pairs)]
        # permute legs such that the ones with the smaller index do not move
        num_codom_legs = tensor.num_codomain_legs
        idcs = remaining[:]
        for idx, pair in zip(insert_idcs, pairs):
            idcs[idx: idx] = list(pair)
            if pair[0] < tensor.num_codomain_legs and pair[1] >= tensor.num_codomain_legs:
                num_codom_legs += 1  # leg at pair[1] is bent up
        num_dom_legs = tensor.num_legs - num_codom_legs

        if not np.all(idcs == np.arange(tensor.num_legs, dtype=int)):
            if tensor.symmetry.braiding_style.value >= 20 and levels is None:
                msg = 'need to specify levels when (implicitly) permuting legs \
                       with non-abelian braiding'
                raise BraidChiralityUnspecifiedError(msg)
            # TODO do we only want to check the levels that are actually needed for the braids?
            if levels is not None:
                for pair in pairs:
                    if levels[pair[0]] is None or levels[pair[1]] is None:
                        continue
                    for i, level in enumerate(levels):
                        if i in pair:
                            continue
                        if level is None:
                            continue
                        if (level < levels[pair[0]]) != (level < levels[pair[1]]):
                            msg = ('inconsistent levels: there should not be a leg with a level '
                                   'between the levels of a pair of legs that is traced over')
                            raise BraidChiralityUnspecifiedError(msg)

        # Build new codomain and domain
        # TODO (JU) this is duplicate code with tensors._permute_legs, but we cant import that
        #           here (cyclic)
        codomain_idcs = idcs[:num_codom_legs]
        domain_idcs = idcs[num_codom_legs:][::-1]
        mixes_codomain_domain = any(i >= tensor.num_codomain_legs for i in codomain_idcs) \
            or any(i < tensor.num_codomain_legs for i in domain_idcs)
        if mixes_codomain_domain:
            codom = TensorProduct([tensor._as_codomain_leg(i) for i in codomain_idcs],
                                  symmetry=tensor.symmetry)
            dom = TensorProduct([tensor._as_domain_leg(i) for i in domain_idcs],
                                symmetry=tensor.symmetry)
        else:
            # (co)domain has the same factor as before, only permuted -> can re-use sectors!
            codom = tensor.codomain.permuted(codomain_idcs)
            dom = tensor.domain.permuted([tensor.num_legs - 1 - i for i in domain_idcs])
        data = self.permute_legs(tensor, codomain_idcs=codomain_idcs, domain_idcs=domain_idcs,
                                 new_codomain=codom, new_domain=dom,
                                 mixes_codomain_domain=mixes_codomain_domain, levels=levels)

        # only consider coupled sectors in data that are consistent with co(domain) after tracing
        coupled_sectors = []
        for _, i in data.block_inds:
            # OPTIMIZE use sorted properties to speed this up.
            sector = dom.sector_decomposition[i]
            if new_domain.sector_decomposition_where(sector) is None:
                continue
            if new_codomain.sector_decomposition_where(sector) is None:
                continue
            coupled_sectors.append(sector)
        new_data = self.zero_data(new_codomain, new_domain, tensor.dtype, tensor.device,
                                  all_blocks=True)
        # block indices
        old_inds = [data.block_ind_from_coupled(c, dom) for c in coupled_sectors]
        new_inds = [new_data.block_ind_from_coupled(c, new_domain) for c in coupled_sectors]

        # step 2: compute new entries: iterate over all trees in the untraced
        # spaces and construct the consistent trees in the traced spaces

        # need to get updated indices after permuting the legs
        codom_unc_idcs = [i for i, idx in enumerate(idcs[:num_codom_legs]) if idx in remaining]
        codom_inner_idcs = [i - 2 for i in codom_unc_idcs[2:]]
        codom_multi_idcs = [i - 1 for i in codom_unc_idcs[1:]]
        codom_tree_idcs = [i for i, idx in enumerate(idcs[:num_codom_legs]) if idx in idcs1]

        dom_unc_idcs = [num_dom_legs - 1 - i for i, idx in enumerate(idcs[num_codom_legs:])
                        if idx in remaining][::-1]
        dom_inner_idcs = [i - 2 for i in dom_unc_idcs[2:]]
        dom_multi_idcs = [i - 1 for i in dom_unc_idcs[1:]]
        dom_tree_idcs = [num_dom_legs - 1 - i for i, idx in enumerate(idcs[num_codom_legs:])
                         if idx in idcs2][::-1]

        tr_idcs = idcs[:num_codom_legs] + idcs[num_codom_legs:][::-1]
        tr_idcs1 = [i for i, idx in enumerate(tr_idcs) if idx in idcs1]
        tr_idcs2 = [i for i, idx in enumerate(tr_idcs) if idx in idcs2]
        remain_idcs = [i for i, idx in enumerate(tr_idcs) if idx in remaining]

        for codom_tree, codom_slc, codom_mults, ind in codom.iter_tree_blocks(coupled_sectors):
            on_diag, factor_codom = _partial_trace_helper(codom_tree, codom_tree_idcs)
            if not on_diag:
                continue
            new_codom_tree = FusionTree(
                tensor.symmetry, codom_tree.uncoupled[codom_unc_idcs], codom_tree.coupled,
                codom_tree.are_dual[codom_unc_idcs], codom_tree.inner_sectors[codom_inner_idcs],
                codom_tree.multiplicities[codom_multi_idcs]
            )
            new_codom_slc = new_codomain.tree_block_slice(new_codom_tree)
            old_ind = old_inds[ind]
            new_ind = new_inds[ind]
            for dom_tree, dom_slc, dom_mults, _ in dom.iter_tree_blocks([codom_tree.coupled]):
                on_diag, factor_dom = _partial_trace_helper(dom_tree, dom_tree_idcs)
                if not on_diag:
                    continue
                tmp_shape = (*codom_mults, *dom_mults)
                new_dom_tree = FusionTree(
                    tensor.symmetry, dom_tree.uncoupled[dom_unc_idcs], dom_tree.coupled,
                    dom_tree.are_dual[dom_unc_idcs], dom_tree.inner_sectors[dom_inner_idcs],
                    dom_tree.multiplicities[dom_multi_idcs]
                )
                new_dom_slc = new_domain.tree_block_slice(new_dom_tree)

                old_block = data.blocks[old_ind][codom_slc, dom_slc]
                old_block = self.block_backend.reshape(old_block, tmp_shape)
                contribution = self.block_backend.trace_partial(old_block, tr_idcs1, tr_idcs2, remain_idcs)
                new_shape = (new_codom_slc.stop - new_codom_slc.start, new_dom_slc.stop - new_dom_slc.start)
                contribution = self.block_backend.reshape(contribution, new_shape)
                contribution *= factor_codom * np.conj(factor_dom)
                new_data.blocks[new_ind][new_codom_slc, new_dom_slc] += contribution
        new_data.discard_zero_blocks(self.block_backend, self.eps)

        if len(remaining) == 0:
            if len(new_data.blocks) == 0:
                return tensor.dtype.zero_scalar, None, None
            elif len(new_data.blocks) == 1:
                return self.block_backend.item(new_data.blocks[0]), None, None
            raise RuntimeError
        return new_data, new_codomain, new_domain

    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     new_codomain: TensorProduct, new_domain: TensorProduct,
                     mixes_codomain_domain: bool, levels: list[int | None]) -> FusionTreeData:
        instructions = permute_legs_instructions(
            num_codomain_legs=a.num_codomain_legs, num_domain_legs=a.num_domain_legs,
            codomain_idcs=codomain_idcs, domain_idcs=domain_idcs, levels=levels,
            has_symmetric_braid=a.symmetry.has_symmetric_braid
        )
        return self.apply_instructions(
            a, instructions, codomain_idcs=codomain_idcs, domain_idcs=domain_idcs,
            new_codomain=new_codomain, new_domain=new_domain,
            mixes_codomain_domain=mixes_codomain_domain
        )

    def qr(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        q_blocks = []
        q_block_inds = []
        r_blocks = []
        r_block_inds = []
        n = 0  # running index, indicating we have already processed a_blocks[:n]
        bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        itr = iter_common_sorted_arrays(a.codomain.sector_decomposition,
                                        a.domain.sector_decomposition)
        for i_new, (i_cod, i_dom) in enumerate(itr):
            q_block_inds.append([i_cod, i_new])
            if bi_cod == i_cod:
                q, r = self.block_backend.matrix_qr(a_blocks[n], full=False)
                q_blocks.append(q)
                r_blocks.append(r)
                r_block_inds.append([i_new, i_dom])
                n += 1
                bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                # there is no block for that sector. => r=0, no need to set it.
                # choose basis vectors for q as standard basis vectors (cols/rows of eye)
                B_cod = a.codomain.multiplicities[i_cod]
                B_new = new_co_domain.multiplicities[i_new]
                q_blocks.append(self.block_backend.eye_matrix(B_cod, a.dtype)[:, :B_new])
        if len(q_block_inds) == 0:
            q_block_inds = np.zeros((0, 2), int)
        else:
            q_block_inds = np.array(q_block_inds)
        if len(r_block_inds) == 0:
            r_block_inds = np.zeros((0, 2), int)
        else:
            r_block_inds = np.array(r_block_inds)
        q_data = FusionTreeData(q_block_inds, q_blocks, a.dtype, a.data.device)
        r_data = FusionTreeData(r_block_inds, r_blocks, a.dtype, a.data.device)
        return q_data, r_data

    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        numbers = []
        blocks = tensor.data.blocks
        block_inds = tensor.data.block_inds
        n = 0
        bi = -1 if n >= len(block_inds) else block_inds[n, 0]
        for i in range(tensor.codomain.num_sectors):
            if i == bi:
                block = blocks[n]
                n += 1
                bi = -1 if n >= len(block_inds) else block_inds[n, 0]
            else:
                block = self.block_backend.zeros([tensor.codomain.multiplicities[n]],
                                                 dtype=tensor.dtype)
            numbers.append(block_func(block))
        return func(numbers)

    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        in_domain, co_domain_idx, leg_idx = a._parse_leg_idx(leg)
        ax_a = int(in_domain)  # 1 if in_domain, 0 else

        a_blocks = a.data.blocks
        b_blocks = b.data.blocks
        a_block_inds = a.data.block_inds
        b_block_inds = b.data.block_inds

        if (in_domain and a.domain.num_factors == 1) or (not in_domain and a.codomain.num_factors == 1):
            # special case where it is essentially compose.
            blocks = []
            block_inds = []

            if len(a_block_inds) > 0 and len(b_block_inds) > 0:
                for n_a, n_b in iter_common_sorted(a_block_inds[:, ax_a], b_block_inds[:, 1 - ax_a]):
                    blocks.append(self.block_backend.scale_axis(a_blocks[n_a], b_blocks[n_b], axis=ax_a))
                    if in_domain:
                        block_inds.append([a_block_inds[n_a, 0], b_block_inds[n_b, 1]])
                    else:
                        block_inds.append([b_block_inds[n_b, 0], a_block_inds[n_a, 0]])
            if len(block_inds) == 0:
                block_inds = np.zeros((0, 2), int)
            else:
                block_inds = np.array(block_inds, int)
            return FusionTreeData(block_inds, blocks, a.dtype, a.data.device)

        iter_space = a.domain if in_domain else a.codomain
        if a.has_pipes:
            # use flattened tensor product -> need to shift co_domain_idx
            for i in range(co_domain_idx):
                co_domain_idx += len(iter_space.flat_leg_idcs(i)) - 1
            iter_space = TensorProduct(
                factors=iter_space.flat_legs(), symmetry=iter_space.symmetry,
                _sector_decomposition=iter_space.sector_decomposition, _multiplicities=iter_space.multiplicities
            )
        blocks = []
        block_inds = np.zeros((0, 2), int)
        # potential coupled sectors
        coupled_sectors = np.array([a.codomain.sector_decomposition[ind[0]] for ind in a_block_inds])
        ind_mapping = {}  # mapping between index in coupled sectors and index in blocks
        for uncoupled, slc, coupled_ind in iter_space.iter_forest_blocks(coupled_sectors):
            ind = a_block_inds[coupled_ind, 1]
            ind_b = b.data.block_ind_from_coupled(uncoupled[co_domain_idx], b.domain)
            if ind_b is None:  # zero block
                continue

            if not ind in block_inds[:, 1]:
                ind_mapping[coupled_ind] = len(blocks)
                block_inds = np.append(block_inds, np.array([[a_block_inds[coupled_ind, 0], ind]]), axis=0)
                shape = self.block_backend.get_shape(a_blocks[coupled_ind])
                blocks.append(self.block_backend.zeros(shape, a.dtype))

            reshape = [iter_space[i].sector_multiplicity(sec) for i, sec in enumerate(uncoupled)]
            if in_domain:
                forest = a_blocks[coupled_ind][:, slc]
                initial_shape = self.block_backend.get_shape(forest)
                # add -1 for reshaping to take care of multiple trees within the same forest
                forest = self.block_backend.reshape(forest, (initial_shape[0], -1, *reshape))
                slcs = [slice(initial_shape[0]), slc]
            else:
                forest = a_blocks[coupled_ind][slc, :]
                initial_shape = self.block_backend.get_shape(forest)
                forest = self.block_backend.reshape(forest, (-1, *reshape, initial_shape[1]))
                slcs = [slc, slice(initial_shape[1])]

            # + 1 for axis comes from adding -1 to the reshaping
            forest = self.block_backend.scale_axis(forest, b_blocks[ind_b], axis=ax_a + co_domain_idx + 1)
            forest = self.block_backend.reshape(forest, initial_shape)
            blocks[ind_mapping[coupled_ind]][slcs[0], slcs[1]] = forest
        return FusionTreeData(block_inds, blocks, a.dtype, a.data.device)

    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int], codomain_split: list[int],
                   domain_split: list[int], new_codomain: TensorProduct, new_domain: TensorProduct
                   ) -> Data:
        raise NotImplementedError('FusionTreeBackend.split_legs not implemented')

    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        return a.data

    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        # supports all symmetries
        return isinstance(symmetry, Symmetry)

    def svd(self, a: SymmetricTensor, new_co_domain: TensorProduct, algorithm: str | None
            ) -> tuple[Data, DiagonalData, Data]:
        a_blocks = a.data.blocks
        a_block_inds = a.data.block_inds
        #
        u_blocks = []
        s_blocks = []
        vh_blocks = []
        u_block_inds = []
        s_block_inds = []
        vh_block_inds = []
        #
        n = 0
        bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
        itr = iter_common_sorted_arrays(a.codomain.sector_decomposition, a.domain.sector_decomposition)
        for i_new, (i_cod, i_dom) in enumerate(itr):
            u_block_inds.append([i_cod, i_new])
            vh_block_inds.append([i_new, i_dom])
            if bi_cod == i_cod:
                u, s, vh = self.block_backend.matrix_svd(a_blocks[n], algorithm=algorithm)
                u_blocks.append(u)
                s_blocks.append(s)
                vh_blocks.append(vh)
                s_block_inds.append([i_new, i_new])
                n += 1
                bi_cod = -1 if n >= len(a_block_inds) else a_block_inds[n, 0]
            else:
                # there is no block for that sector. => s=0, no need to set it.
                # choose basis vectors for u/vh as standard basis vectors (cols/rows of eye)
                B_cod = a.codomain.multiplicities[i_cod]
                B_dom = a.domain.multiplicities[i_dom]
                B_new = new_co_domain.multiplicities[i_new]
                u_blocks.append(self.block_backend.eye_matrix(B_cod, a.dtype)[:, :B_new])
                vh_blocks.append(self.block_backend.eye_matrix(B_dom, a.dtype)[:B_new, :])
        if len(u_block_inds) == 0:
            u_block_inds = np.zeros((0, 2), int)
        else:
            u_block_inds = np.array(u_block_inds, int)
        if len(s_block_inds) == 0:
            s_block_inds = np.zeros((0, 2), int)
        else:
            s_block_inds = np.array(s_block_inds, int)
        if len(vh_block_inds) == 0:
            vh_block_inds = np.zeros((0, 2), int)
        else:
            vh_block_inds = np.array(vh_block_inds, int)
        u_data = FusionTreeData(u_block_inds, u_blocks, a.dtype, a.data.device)
        s_data = FusionTreeData(s_block_inds, s_blocks, a.dtype.to_real, a.data.device)
        vh_data = FusionTreeData(vh_block_inds, vh_blocks, a.dtype, a.data.device)
        return u_data, s_data, vh_data

    def state_tensor_product(self, state1: Block, state2: Block, pipe: LegPipe):
        # TODO clearly define what this should do in tensors.py first!
        raise NotImplementedError

    def to_dense_block(self, a: SymmetricTensor) -> Block:
        assert a.symmetry.can_be_dropped
        if a.has_pipes:
            return a._to_dense_block_by_splitting_pipes()
        J = len(a.codomain.factors)
        K = len(a.domain.factors)
        num_legs = J + K
        dtype = Dtype.common(a.data.dtype, a.symmetry.fusion_tensor_dtype)
        sym = a.symmetry
        # build in internal basis order, is converted to public basis order in SymmetricTensor.to_dense_block
        # build in codomain/domain leg order first, then permute legs in the end
        # [i1,...,iJ,j1,...,jK]
        shape = [leg.dim for leg in a.codomain.factors] + [leg.dim for leg in a.domain.factors]
        res = self.block_backend.zeros(shape, dtype)
        for bi_cod, block in zip(a.data.block_inds[:, 0], a.data.blocks):
            coupled = a.codomain.sector_decomposition[bi_cod]
            i1 = 0  # start row index of the current forest block
            i2 = 0  # start column index of the current forest block
            for b_sectors, n_dims, j2 in a.domain.iter_uncoupled(yield_slices=True):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = a.domain.tree_block_size(b_sectors)
                for a_sectors, m_dims, j1 in a.codomain.iter_uncoupled(yield_slices=True):
                    a_dims = sym.batch_sector_dim(a_sectors)
                    tree_block_height = a.codomain.tree_block_size(a_sectors)
                    entries, num_alpha_trees, num_beta_trees = self._get_forest_block_contribution(
                        block, sym, a.codomain, a.domain, coupled, a_sectors, b_sectors,
                        a_dims, b_dims, tree_block_width, tree_block_height, i1, i2, m_dims, n_dims,
                        dtype
                    )
                    forest_b_height = num_alpha_trees * tree_block_height
                    forest_b_width = num_beta_trees * tree_block_width
                    if forest_b_height == 0 or forest_b_width == 0:
                        continue
                    # entries : [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
                    # permute to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
                    perm = [i + offset for i in range(num_legs) for offset in [0, num_legs]]
                    entries = self.block_backend.permute_axes(entries, perm)
                    # reshape to [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
                    shape = [d_a * m for d_a, m in zip(a_dims, m_dims)] + \
                        [d_b * n for d_b, n in zip(b_dims, n_dims)]
                    entries = self.block_backend.reshape(entries, shape)
                    res[(*j1, *j2)] += entries
                    i1 += forest_b_height  # move down by one forest-block
                i1 = 0  # reset to the top of the block
                i2 += forest_b_width  # move right by one forest-block
        # permute leg order [i1,...,iJ,j1,...,jK] -> [i1,...,iJ,jK,...,j1]
        res = self.block_backend.permute_axes(res, [*range(J), *reversed(range(J, J + K))])
        return res

    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        raise NotImplementedError('to_dense_block_trivial_sector not implemented')

    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> FusionTreeData:
        blocks = [self.block_backend.to_dtype(block, dtype) for block in a.data.blocks]
        return FusionTreeData(a.data.block_inds, blocks, dtype, a.data.device)

    def trace_full(self, a: SymmetricTensor) -> float | complex:
        return sum(
            (a.codomain.sector_qdims[bi_cod] * self.block_backend.trace_full(block)
             for bi_cod, block in zip(a.data.block_inds[:, 0], a.data.blocks)),
            a.dtype.zero_scalar
        )

    def transpose(self, a: SymmetricTensor, new_codomain: TensorProduct, new_domain: TensorProduct
                  ) -> Data:
        #                │            ╭─╮     │
        #    ╭─────╮     │            ╰─│─────│──╮     <- at both crossings, vertical │ is on top
        #    │  ┏━━┷━━┓  │           ┏━━┷━━┓  │  │
        #    │  ┃  Y  ┃  │           ┃  Y  ┃  │  │
        #    │  ┗━━┯━━┛  │           ┗━━┯━━┛  │  │
        #    │     │     │     =        │     │  │
        #    │  ┏━━┷━━┓  │           ┏━━┷━━┓  │  │
        #    │  ┃  X  ┃  │           ┃  X  ┃  │  │
        #    │  ┗━━┯━━┛  │           ┗━━┯━━┛  │  │
        #    │     ╰─────╯              ╰─────╯  │
        twist_instruction = TwistInstruction(codomain=True, idcs=[*range(a.num_codomain_legs)],
                                             overtwist=False)
        codomain_idcs = list(range(a.num_codomain_legs, a.num_legs))
        domain_idcs = list(reversed(range(a.num_codomain_legs)))
        levels = list(range(a.num_legs))  # codomain under domain
        permute_instructions = permute_legs_instructions(
            num_codomain_legs=a.num_codomain_legs, num_domain_legs=a.num_domain_legs,
            codomain_idcs=codomain_idcs, domain_idcs=domain_idcs, levels=levels,
            has_symmetric_braid=a.symmetry.has_symmetric_braid
        )
        instructions = [twist_instruction, *permute_instructions]
        return self.apply_instructions(a, instructions=instructions, codomain_idcs=codomain_idcs,
                                       domain_idcs=domain_idcs, new_codomain=new_codomain,
                                       new_domain=new_domain, mixes_codomain_domain=True)

    def truncate_singular_values(self, S: DiagonalTensor, chi_max: int | None, chi_min: int,
                                 degeneracy_tol: float, trunc_cut: float, svd_min: float
                                 ) -> tuple[MaskData, ElementarySpace, float, float]:
        # build a numpy array of the singular values and a numpy array of the qdims
        num_singular_values = np.sum(S.leg.multiplicities)
        S_np = np.zeros(num_singular_values, S.dtype.to_numpy_dtype())
        qdims = np.empty(num_singular_values, float)
        slices = []
        stop = 0
        i = 0  # have already considered blocks[:i]
        S_blocks = S.data.blocks
        S_block_inds = S.data.block_inds
        S_num_blocks = len(S_blocks)
        for j, (qdim, mult) in enumerate(zip(S.domain.sector_qdims, S.domain.multiplicities)):
            start = stop
            stop += mult
            slc = slice(start, stop)
            slices.append(slc)
            if i < S_num_blocks and S_block_inds[i, 0] == j:  # we have a block for that coupled sector
                S_np[slc] = self.block_backend.to_numpy(S.data.blocks[i])
                i += 1
            qdims[slc] = qdim

        # select which to keep
        keep, err, new_norm = self._truncate_singular_values_selection(
            S=S_np, qdims=qdims, chi_max=chi_max, chi_min=chi_min, degeneracy_tol=degeneracy_tol,
            trunc_cut=trunc_cut, svd_min=svd_min
        )

        # build the Mask
        if S.leg._basis_perm is not None:
            # TODO not sure how to deal with the basis perm here...
            #      but ideally the new leg of an SVD has no basis_perm anyway
            raise NotImplementedError
        
        large_leg_block_inds = []
        mask_blocks = []
        small_leg_sectors = []
        small_leg_multiplicities = []
        for i, (slc, sector) in enumerate(zip(slices, S.domain.sector_decomposition)):
            block = keep[slc]
            if not np.any(block):
                continue  # all False. skip this block.
            large_leg_block_inds.append(i)
            mask_blocks.append(self.block_backend.block_from_numpy(block))
            small_leg_sectors.append(sector)
            small_leg_multiplicities.append(np.sum(block))
        #
        mask_block_inds = np.column_stack([np.arange(len(small_leg_sectors)), large_leg_block_inds])
        small_leg_sectors = np.array(small_leg_sectors, int)
        small_leg_multiplicities = np.array(small_leg_multiplicities, int)
        #
        mask_data = FusionTreeData(mask_block_inds, mask_blocks, dtype=Dtype.bool,
                                   device=S.data.device, is_sorted=True)
        small_leg = ElementarySpace.from_sector_decomposition(
            S.symmetry, small_leg_sectors, small_leg_multiplicities, is_dual=S.leg.is_dual
        )
        small_leg._basis_perm = None  # OPTIMIZE avoid computing it, if we reset it anyway
        small_leg._inverse_basis_perm = None
        return mask_data, small_leg, err, new_norm
        
    def zero_data(self, codomain: TensorProduct, domain: TensorProduct, dtype: Dtype, device: str,
                  all_blocks: bool = False) -> FusionTreeData:
        if not all_blocks:
            return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=dtype,
                                  device=device)
            
        block_shapes = []
        block_inds = []
        for j, coupled in enumerate(domain.sector_decomposition):
            i = codomain.sector_decomposition_where(coupled)
            if i is None:
                continue
            shp = (codomain.block_size(i), domain.block_size(j))
            block_shapes.append(shp)
            block_inds.append([i, j])

        if len(block_inds) == 0:
            return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=dtype,
                                  device=device)
        
        block_inds = np.array(block_inds)
        zero_blocks = [self.block_backend.zeros(block_shape, dtype=dtype)
                       for block_shape in block_shapes]
        return FusionTreeData(block_inds, zero_blocks, dtype=dtype, device=device, is_sorted=True)

    def zero_diagonal_data(self, co_domain: TensorProduct, dtype: Dtype, device: str
                           ) -> DiagonalData:
        return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=dtype,
                              device=device)

    def zero_mask_data(self, large_leg: Space, device: str,) -> MaskData:
        return FusionTreeData(block_inds=np.zeros((0, 2), int), blocks=[], dtype=Dtype.bool,
                              device=device)

    # INTERNAL FUNCTIONS

    def _get_forest_block_contribution(self, block, sym: Symmetry, codomain, domain, coupled,
                                       a_sectors, b_sectors, a_dims, b_dims, tree_block_width,
                                       tree_block_height, i1_init, i2_init, m_dims, n_dims,
                                       dtype):
        """Helper function for :meth:`to_dense_block`.

        Obtain the contributions from a given forest block

        Parameters
        ----------
        block:
            The current block
        sym:
            The symmetry
        codomain, domain:
            The codomain and domain of the new tensor
        coupled, dim_c:
            The coupled sector of the current block and its quantum dimension
        a_sectors:
            The codomain uncoupled sectors [a1, a2, ..., aJ]
        b_sectors:
            The domain uncoupled sectors [b1, b2, ..., bK]
        tree_block_width:
            Equal to ``tree_block_size(domain, b_sectors)``
        tree_block_height:
            Equal to ``tree_block_size(codomain, a_sectors)``
        i1_init, i2_init:
            The start indices of the current forest block within the block

        Returns
        -------
        entries:
            The entries of the dense block corresponding to the given uncoupled sectors.
            Legs [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
        num_alpha_trees:
            The number of fusion trees from ``a_sectors`` to ``coupled``
        num_beta_trees:
            The number of fusion trees from ``b_sectors`` to ``coupled``
        """
        # OPTIMIZE do one loop per vertex in the tree instead.
        i1 = i1_init  # i1: start row index of the current tree block within the block
        i2 = i2_init  # i2: start column index of the current tree block within the block
        # TODO should probably replace (co)domain.factors with a flat list of ElementarySpace !!
        alpha_tree_iter = fusion_trees(sym, a_sectors, coupled, [sp.is_dual for sp in codomain.factors])
        beta_tree_iter = fusion_trees(sym, b_sectors, coupled, [sp.is_dual for sp in domain.factors])
        entries = self.block_backend.zeros([*a_dims, *b_dims, *m_dims, *n_dims], dtype)
        for alpha_tree in alpha_tree_iter:
            Y = self.block_backend.conj(alpha_tree.as_block(backend=self))  # [a1,...,aJ,c]
            for beta_tree in beta_tree_iter:
                X = beta_tree.as_block(backend=self)  # [b1,...,bK,c]
                symmetry_data = self.block_backend.tdot(Y, X, [-1], [-1])  # [a1,...,aJ,b1,...,bK]
                idx1 = slice(i1, i1 + tree_block_height)
                idx2 = slice(i2, i2 + tree_block_width)
                degeneracy_data = block[idx1, idx2]  # [M, N]
                # [M, N] -> [m1,...,mJ,n1,...,nK]
                degeneracy_data = self.block_backend.reshape(degeneracy_data, [*m_dims, *n_dims])
                entries += self.block_backend.outer(symmetry_data, degeneracy_data)  # [{aj} {bk} {mj} {nk}]
                i2 += tree_block_width
            i2 = i2_init  # reset to the left of the current forest-block
            i1 += tree_block_height
        # OPTIMIZE count loop iterations above instead?  (same in _add_forest_block_entries)
        num_alpha_trees = len(alpha_tree_iter)
        num_beta_trees = len(beta_tree_iter)
        return entries, num_alpha_trees, num_beta_trees

    def _add_forest_block_entries(self, block, entries, sym: Symmetry, codomain: TensorProduct,
                                  domain: TensorProduct, coupled, dim_c, a_sectors, b_sectors,
                                  tree_block_width, tree_block_height, i1_init, i2_init):
        """Helper function for :meth:`from_dense_block`.

        Adds the entries from a single forest-block to the current `block`, in place.

        Parameters
        ----------
        block:
            The block to modify
        entries:
            The entries of the dense block corresponding to the given uncoupled sectors.
            Legs [a1,...,aJ, b1,...,bK, m1,...,mJ, n1,...,nK]
        sym:
            The symmetry
        codomain, domain:
            The codomain and domain of the new tensor
        coupled, dim_c:
            The coupled sector of the current block and its quantum dimension
        a_sectors:
            The codomain uncoupled sectors [a1, a2, ..., aJ]
        b_sectors:
            The domain uncoupled sectors [b1, b2, ..., bK]
        tree_block_width:
            Equal to ``tree_block_size(domain, b_sectors)``
        tree_block_height:
            Equal to ``tree_block_size(codomain, a_sectors)``
        i1_init, i2_init:
            The start indices of the current forest block within the block

        Returns
        -------
        num_alpha_trees:
            The number of fusion trees from ``a_sectors`` to ``coupled``
        num_beta_trees :
            The number of fusion trees from ``b_sectors`` to ``coupled``
        """
        # OPTIMIZE do one loop per vertex in the tree instead.
        i1 = i1_init  # i1: start row index of the current tree block within the block
        i2 = i2_init  # i2: start column index of the current tree block within the block
        domain_are_dual = [sp.is_dual for sp in domain.factors]
        codomain_are_dual = [sp.is_dual for sp in codomain.factors]
        J = codomain.num_factors
        K = domain.num_factors
        range_J = list(range(J))  # used in tdot calls below
        range_K = list(range(K))  # used in tdot calls below
        range_JK = list(range(J + K))
        alpha_tree_iter = fusion_trees(sym, a_sectors, coupled, codomain_are_dual)
        beta_tree_iter = fusion_trees(sym, b_sectors, coupled, domain_are_dual)
        for alpha_tree in alpha_tree_iter:
            X = alpha_tree.as_block(backend=self)
            # entries: [a1,...,aJ,b1,...,bK,m1,...,mJ,n1,...,nK]
            X_projected = self.block_backend.tdot(entries, X, range_J, range_J)  # [{bk}, {mj}, {nk}, c]
            for beta_tree in beta_tree_iter:
                Y = self.block_backend.conj(beta_tree.as_block(backend=self))
                XY_projected = self.block_backend.tdot(X_projected, Y, range_K, range_K)  # [{mj}, {nk}, c, c']
                # projected onto the identity on [c, c']
                tree_block = self.block_backend.trace_partial(XY_projected, [-2], [-1], range_JK) / dim_c
                # [m1,...,mJ,n1,...,nK] -> [M, N]
                ms_ns = self.block_backend.get_shape(tree_block)
                shape = (prod(ms_ns[:J]), prod(ms_ns[J:]))
                tree_block = self.block_backend.reshape(tree_block, shape)
                idx1 = slice(i1, i1 + tree_block_height)
                idx2 = slice(i2, i2 + tree_block_width)
                # make sure we set in-range elements! otherwise item assignment silently does nothing.
                assert 0 <= idx1.start < idx1.stop <= block.shape[0]
                assert 0 <= idx2.start < idx2.stop <= block.shape[1]
                block[idx1, idx2] = tree_block
                i2 += tree_block_width  # move right by one tree-block
            i2 = i2_init  # reset to the left of the current forest-block
            i1 += tree_block_height  # move down by one tree-block (we reset to the left at start of the loop)
        num_alpha_trees = len(alpha_tree_iter)  # OPTIMIZE count loop iterations above instead?
        num_beta_trees = len(beta_tree_iter)
        return num_alpha_trees, num_beta_trees


class Instruction(metaclass=ABCMeta):
    """An instruction represents an elementary operation on a tensor.

    This is e.g. a single NN-braid or a single leg bend, for which we have symbols (R, C, B)
    that tell us exactly how the fusion tree(-pair)s within a tensor transform.

    Even though the base class currently does not do anything, we keep it around for type checking
    etc. and may add functionality in the future.

    We can then build more general tensor operations from these instructions,
    see e.g. :meth:`FusionTreeBackend.permute_legs`.
    """

    pass


@dataclass(frozen=True, slots=True)
class BraidInstruction(Instruction):
    """Instruction to braid two neighboring legs.

    Attributes
    ----------
    codomain : bool
        If the braid is in the codomain, otherwise in the domain.
    idx : int
        Which leg of the (co-)domain braids.
        We braid ``(co)domain[idx]`` with ``(co)domain[idx + 1]``
    overbraid : bool
        Specifies the chirality of the braid. An overbraid is a braid where the leg that goes
        from bottom left to top right is on top, see notes below.

    Notes
    -----
    Examples for over-braids::

        |    │    ╲ ╱    │                      │   │   │   │
        |    │     ╱     │                     ┏┷━━━┷━━━┷━━━┷┓
        |    │    ╱ ╲    │                     ┃             ┃
        |   ┏┷━━━┷━━━┷━━━┷┓                    ┗━━┯━━━┯━━━┯━━┛
        |   ┃             ┃         OR             ╲ ╱    │
        |   ┗━━┯━━━┯━━━┯━━┛                         ╱     │
        |      │   │   │                           ╱ ╲    │

    Examples for under-braids::

        |    │    ╲ ╱    │                      │   │   │   │
        |    │     ╲     │                     ┏┷━━━┷━━━┷━━━┷┓
        |    │    ╱ ╲    │                     ┃             ┃
        |   ┏┷━━━┷━━━┷━━━┷┓                    ┗━━┯━━━┯━━━┯━━┛
        |   ┃             ┃         OR             ╲ ╱    │
        |   ┗━━┯━━━┯━━━┯━━┛                         ╲     │
        |      │   │   │                           ╱ ╲    │

    """

    codomain: bool
    idx: int
    overbraid: bool


@dataclass(frozen=True, slots=True)
class BendInstruction(Instruction):
    """Instruction to bend the rightmost leg of the codomain down (of the domain up)."""

    bend_up: bool


@dataclass(frozen=True, slots=True)
class TwistInstruction(Instruction):
    """Instruction to apply a twist on one leg.

    Attributes
    ----------
    codomain : bool
        If the twist is in the codomain, otherwise in the domain.
    idcs : list of int
        Which legs of the (co-)domain are twisted; we twist ``(co)domain[idcs]``.
        Must be contiguous.
    overtwist : bool
        Specifies the chirality of the twist. An overtwist (undertwist) has an overbraid
        (underbraid) at the center, and a cup and cap.

    Notes
    -----
    Let us first illustrate how the chirality is given by :attr:`overtwist`.
    For simplicity, we always show ``idcs=[-1]``.
    Example for over-twists::

        |    │   │   │   │   ╭─╮             │   │   │   │
        |    │   │   │    ╲ ╱  │            ┏┷━━━┷━━━┷━━━┷┓
        |    │   │   │     ╱   │            ┃             ┃
        |    │   │   │    ╱ ╲  │            ┗━━┯━━━┯━━━┯━━┛╭─╮
        |   ┏┷━━━┷━━━┷━━━┷┓  ╰─╯               │   │    ╲ ╱  │
        |   ┃             ┃         OR         │   │     ╱   │
        |   ┗━━┯━━━┯━━━┯━━┛                    │   │    ╱ ╲  │
        |      │   │   │                       │   │   │   ╰─╯

    Examples for under-twists::

        |    │   │   │   │   ╭─╮             │   │   │   │
        |    │   │   │    ╲ ╱  │            ┏┷━━━┷━━━┷━━━┷┓
        |    │   │   │     ╲   │            ┃             ┃
        |    │   │   │    ╱ ╲  │            ┗━━┯━━━┯━━━┯━━┛╭─╮
        |   ┏┷━━━┷━━━┷━━━┷┓  ╰─╯               │   │    ╲ ╱  │
        |   ┃             ┃         OR         │   │     ╲   │
        |   ┗━━┯━━━┯━━━┯━━┛                    │   │    ╱ ╲  │
        |      │   │   │                       │   │   │   ╰─╯

    For multiple legs (``len(idcs) > 1``), we twist them together, e.g.::

        |
        |
        |    │   │   │   │   ╭──────╮
        |    │   │    ╲   ╲ ╱       │
        |    │   │     ╲   ╱   ╭─╮  │
        |    │   │      ╲ ╱ ╲ ╱  │  │
        |    │   │       ╱   ╱   │  │
        |    │   │      ╱ ╲ ╱ ╲  │  │
        |    │   │     ╱   ╱   ╰─╯  │
        |    │   │    ╱   ╱ ╲       │
        |   ┏┷━━━┷━━━┷━━━┷┓  ╰──────╯
        |   ┃             ┃
        |   ┗━━┯━━━┯━━━┯━━┛
        |      │   │   │
    """

    codomain: bool
    idcs: list[int]
    overtwist: bool


def permute_legs_instructions(num_codomain_legs: int, num_domain_legs: int,
                              codomain_idcs: list[int], domain_idcs: list[int],
                              levels: list[int | None], has_symmetric_braid: bool,
                              ) -> list[Instruction]:
    """Helper to decompose a ``permute_legs`` call into elementary instructions.

    Parameters
    ----------
    num_codomain_legs, num_domain_legs : int
        Number of (co-)domain legs of the input tensor.
    codomain_idcs, domain_idcs : list of int
        ``(co)domain_idcs[i] == j`` means that the leg ``tensor.legs[j]`` should end up at
        ``result.(co)domain[i]``.
    levels : list of {int | None}
        The levels that specify braid chirality.
    has_symmetric_braid : bool
        If the symmetry has a symmetric braid, i.e. if the `levels` are irrelevant.

    Returns
    -------
    instructions : list of Instruction
        A sequence of instructions, such that if applied to a tensor in this order,
        the target permutation is realized.
    """
    instructions = []
    num_legs = num_codomain_legs + num_domain_legs
    # we update levels in-place, to account for swaps: guarantee copy
    levels = list(levels)

    # 0) identify which legs need to be bent
    #   bend_up: legs that are in domain but go to codomain. Sorted by current position in tensor.legs
    bend_up = sorted([i for i in codomain_idcs if i >= num_codomain_legs])
    num_bend_up = len(bend_up)
    stay_up = sorted([i for i in codomain_idcs if i < num_codomain_legs])
    num_stay_up = len(stay_up)
    #   bend_down: are in codomain but go to domain. Sorted by current position in tensor.legs
    bend_down = sorted([i for i in domain_idcs if i < num_codomain_legs])
    num_bend_down = len(bend_down)
    # define lookup tables for those lists, since we want ``bend_up.index(i)`` etc multiple times
    bend_up_lookup = {v: idx for idx, v in enumerate(bend_up)}
    stay_up_lookup = {v: idx for idx, v in enumerate(stay_up)}
    bend_down_lookup = {v: idx for idx, v in enumerate(bend_down)}

    # 1) swap the `bend_down` legs to the very right of the codomain, preserving their order
    #       start with the rightmost of them and move it to the very right
    #       then second rightmost and so on
    for i, leg in enumerate(reversed(bend_down)):  # reverse: go from right to left
        for j in range(leg, num_codomain_legs - 1 - i):
            overbraid = compare_levels(levels[j], levels[j + 1], has_symmetric_braid)
            instructions.append(BraidInstruction(codomain=True, idx=j, overbraid=overbraid))
            levels[j], levels[j + 1] = levels[j + 1], levels[j]
    # TODO put the stay_up into the correct places already at this point?

    # 2) bend them down
    for j in range(num_bend_down):
        instructions.append(BendInstruction(bend_up=False))
    num_codomain_legs -= num_bend_down
    num_domain_legs += num_bend_down

    # 3) permute legs in the domain:
    #       move the `bend_down` leg to the very right, preserving their order
    #       move the other legs of the domain to their target positions
    # 3a) first, construct the permutation that we need to do on the domain.
    domain_perm = []
    for original_leg_idx in domain_idcs:
        if original_leg_idx in bend_down:
            # bend_down[0] got bent down to the very end of the current domain, i.e. -1
            # bend_down[1] to -2, and so on
            curr_domain_idx = num_domain_legs - 1 - bend_down_lookup[original_leg_idx]
        else:
            # so far, we only bend some legs down, to the end of the domain.
            # for these legs, that already were in the domain to begin with, their domain idx is still valid
            curr_domain_idx = num_legs - 1 - original_leg_idx
        domain_perm.append(curr_domain_idx)
    # remaining: those legs that are not in domain_idcs, i.e. exactly the `bend_up`.
    # we permute them to the end of the domain, in order of their appearance in the current domain
    domain_perm.extend(i for i in range(num_domain_legs) if i not in domain_perm)

    # 3b) Now, do this permutation
    for j_domain in permutation_as_swaps(domain_perm):
        # should swap j_domain with j_domain + 1, i.e. j_leg_idx with j_leg_idx - 1
        j_leg_idx = num_legs - 1 - j_domain
        overbraid = compare_levels(levels[j_leg_idx - 1], levels[j_leg_idx], has_symmetric_braid)
        instructions.append(BraidInstruction(codomain=False, idx=j_domain, overbraid=overbraid))
        levels[j_leg_idx], levels[j_leg_idx - 1] = levels[j_leg_idx - 1], levels[j_leg_idx]

    # 4) Bend up
    for j in range(num_bend_up):
        instructions.append(BendInstruction(bend_up=True))
    num_codomain_legs += num_bend_up
    num_domain_legs -= num_bend_up
    assert num_codomain_legs == len(codomain_idcs)
    assert num_domain_legs == len(domain_idcs)

    # 5) permute the legs in the codomain
    # 5a) first build the permutation
    codomain_perm = []
    for original_leg_idx in codomain_idcs:
        if original_leg_idx in bend_up:
            # bend_up[-1] is now the last leg in the codomain
            # bend_up[0] is at position num_stay_up
            codomain_perm.append(num_stay_up + bend_up_lookup[original_leg_idx])
        else:
            codomain_perm.append(stay_up_lookup[original_leg_idx])
    # 5b) now do this permutation
    for j in permutation_as_swaps(codomain_perm):
        overbraid = compare_levels(levels[j], levels[j + 1], has_symmetric_braid)
        instructions.append(BraidInstruction(codomain=True, idx=j, overbraid=overbraid))
        levels[j], levels[j + 1] = levels[j + 1], levels[j]

    return instructions


class TensorMapping(metaclass=ABCMeta):
    r"""Symbolic representation of a map on tensors, defined by the action on tree pairs.

    Note that we dont always represent the whole map, only the components that are actually needed.
    E.g. if we do ``permute_legs`` to a tensor, we only represent the action of the permutation
    on the tree-pairs that actually occur in the tensor.

    This is a base class that defines the common interface.
    See :class:`TreePairMapping` and :class:`IndividualTreeMapping` for the concrete classes.

    Notes
    -----
    Let indices ``I, J, ...`` each label a pair (X_I, Y_I) of a fusion tree X_I and a compatible
    splitting tree Y_I, i.e. it fixes uncoupled sectors, all internal labels of both trees and the
    coupled sector, i.e. it labels a tree block.
    Then with indices ``m`` for the uncoupled multiplicities (labelling entries within a tree block),
    we have the decomposition of tensors as ``T = \sum_{Im} T_{Im} Y_I @ X_I``.
    Now if we apply a linear operation ``f`` (e.g. a braid), we find::

        f(T) = \sum_{Im} T_{Im} f(Y_I @ X_I)
             = \sum_{Im} T_{Im} \sum_J f_{JI} Y_J @ X_J
             = \sum_J ( \sum_I f_{JI} T_{Im} ) Y_J @ X_J

    where ``f_{JI} = <Y_J @ X_J | f(Y_I @ X_I)>`` are the coefficients of ``f`` in the basis of
    tree pairs. This means the blocks of the result are given by

        f(T)_{Jm} = \sum_I f_{JI} T_{Im}

    i.e. are linear combinations of the blocks of T according to the transposed coefficients.
    """

    @classmethod
    def from_instructions(cls, instructions: Iterable[Instruction], codomain: TensorProduct,
                          domain: TensorProduct, block_inds: np.ndarray | None = None
                          ) -> TensorMapping:
        res = cls.from_identity(codomain=codomain, domain=domain, block_inds=block_inds)
        for i in instructions:
            res = res.pre_compose_instruction(i)
        return res

    # METHODS

    def pre_compose_instruction(self, instruction: Instruction, prune_tol: float | None = 1e-15
                                ) -> TensorMapping:
        """Include the action of an instruction, acting as a last step."""
        if isinstance(instruction, BendInstruction):
            res = self.pre_compose_bend_instruction(instruction)
        elif isinstance(instruction, BraidInstruction):
            res = self.pre_compose_braid_instruction(instruction)
        elif isinstance(instruction, TwistInstruction):
            res = self.pre_compose_twist_instruction(instruction)
        elif isinstance(instruction, Instruction):
            raise NotImplementedError
        else:
            raise TypeError
        if prune_tol is not None:
            res.prune(prune_tol)
        return res

    # ABSTRACT

    @classmethod
    @abstractmethod
    def from_identity(cls, codomain: TensorProduct, domain: TensorProduct,
                      block_inds: np.ndarray | None = None
                      ) -> TensorMapping:
        r"""The identity mapping.

        Parameters
        ----------
        codomain, domain : TensorProduct
            The codomain and domain that determine the possible fusion and splitting trees.
        block_inds : 2D array
            Same format and meaning as the :attr:`FusionTreeData.block_inds`.
            If given, we only initialize those components ``Y_I @ X_I -> Y_I @ X_I``
            where the coupled sector of the tree-pair is pointed to by a row in the `block_inds`,
            i.e. if we have ``coupled == codomain.sector_decomposition[block_inds[some_idx, 0]]``.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_compose_bend_instruction(self, instruction: BendInstruction) -> TensorMapping:
        """Special case of :meth:`pre_compose_instruction`."""
        ...

    @abstractmethod
    def pre_compose_braid_instruction(self, instruction: BraidInstruction) -> TensorMapping:
        """Special case of :meth:`pre_compose_instruction`."""
        ...

    @abstractmethod
    def pre_compose_twist_instruction(self, instruction: TwistInstruction) -> TensorMapping:
        """Special case of :meth:`pre_compose_instruction`."""
        ...

    @abstractmethod
    def prune(self, tol: float = 1e-15):
        """Remove small contributions with ``abs(coefficient) < tol`` in-place."""
        ...

    @abstractmethod
    def transform_tensor(self, data: FusionTreeData,
                         codomain: TensorProduct, domain: TensorProduct,
                         new_codomain: TensorProduct, new_domain: TensorProduct,
                         codomain_idcs: list[int], domain_idcs: list[int],
                         block_backend: BlockBackend) -> FusionTreeData:
        r"""Transform a tensor by applying the mapping to its tree-pairs. See class docstring.

        Parameters
        ----------
        data : FusionTreeData
            The data of the input tensor.
        codomain, domain : TensorProduct
            The (co)domain of the input tensor.
        new_codomain, new_domain : TensorProduct
            The (co)domain of the output tensor.
        codomain_idcs, domain_idcs : list of int
            The permutations such that ``new_(co)domain[i] = old_legs[(co)domain_idcs[i]]``.
            This permutation acts on the uncoupled multiplicity indices.
        """
        ...


class TreePairMapping(TensorMapping):
    r"""A :class:`TensorMapping`, defined at the level of tree-pairs, i.e. the general case.

    We store the component ``f_{JI} = <Y_J @ X_J | f(Y_I @ X_I)>``,
    which represents ``Y_I @ X_I \mapsto f_{JI} Y_J @ X_J`` as ``mapping[I][J] = f_{JI}``.
    In practice, the keys are ``I = (Y_I, X_I)`` tuples of two FusionTrees.
    """

    def __init__(self, mapping: SparseMapping[tuple[FusionTree, FusionTree]]):
        self.mapping: SparseMapping[tuple[FusionTree, FusionTree]] = mapping

    @classmethod
    def from_identity(cls, codomain: TensorProduct, domain: TensorProduct,
                      block_inds: np.ndarray | None = None):
        if block_inds is None:
            block_inds = iter_common_sorted_arrays(codomain.sector_decomposition,
                                                   domain.sector_decomposition)
        keys = []
        for i, _ in block_inds:
            coupled = codomain.sector_decomposition[i]
            for Y, *_ in codomain.iter_tree_blocks([coupled]):
                for X, *_ in domain.iter_tree_blocks([coupled]):
                    keys.append((Y, X))
        mapping = SparseMapping[tuple[FusionTree, FusionTree]].from_identity(keys)
        return cls(mapping)

    def test_sanity(self):
        for (Y_i, X_i), self_i in self.mapping.items():
            Y_i.test_sanity()
            X_i.test_sanity()
            assert np.all(X_i.coupled == Y_i.coupled)
            for Y_j, X_j in self_i.keys():
                Y_j.test_sanity()
                X_j.test_sanity()
                assert np.all(X_j.coupled == Y_j.coupled)

    def pre_compose_braid_instruction(self, instruction: BraidInstruction):
        braid_mapping = SparseMapping[FusionTree]()
        if instruction.codomain:
            # the splitting tree in the codomain is represented by a FusionTree and::
            # res_fusion_tree = dagger(res_splitting_tree)
            #                 = dagger(braid(splitting_tree))
            #                 = opposite_braid(dagger(splitting_tree))
            #                 = opposite_braid(fusion_tree)
            # additionally, since we represent t = dagger(t_fusion), coefficients get a conj
            #   a t + b t2 = dagger(conj(a) t_fusion + conj(b) t2_fusion)
            for Y in set(Y for Y, X in self.mapping.nonzero_rows()):
                braid_mapping[Y] = Y.braid(j=instruction.idx, overbraid=not instruction.overbraid,
                                           do_conj=True)
            return self.pre_compose_splitting_tree_mapping(braid_mapping)
        else:
            for X in set(X for Y, X in self.mapping.nonzero_rows()):
                braid_mapping[X] = X.braid(j=instruction.idx, overbraid=instruction.overbraid)
            return self.pre_compose_fusion_tree_mapping(braid_mapping)

    def pre_compose_bend_instruction(self, instruction: BendInstruction):
        bend_mapping = SparseMapping[tuple[FusionTree, FusionTree]]()
        # to pre-compose the bend_mapping, we only need to compute the ``bend_mapping[j][i]``
        # for those ``j`` for which an entry ``self.mapping[k][j]`` exists.
        for Y, X in self.mapping.nonzero_rows():
            bend_mapping[Y, X] = FusionTree.bend_leg(Y, X, instruction.bend_up)
        mapping = self.mapping.pre_compose(bend_mapping)
        return TreePairMapping(mapping)

    def pre_compose_fusion_tree_mapping(self, mapping: SparseMapping[FusionTree]) -> TreePairMapping:
        """Pre-compose with a mapping that acts only on the fusion-trees."""
        res = SparseMapping[tuple[FusionTree, FusionTree]]()
        for k, self_k in self.mapping.items():
            res[k] = res_k = {}
            for (Y, X_j), self_jk in self_k.items():
                for X_i, other_ij in mapping[X_j].items():
                    i = (Y, X_i)
                    res_k[i] = res_k.get(i, 0) + other_ij * self_jk
        return TreePairMapping(res)

    def pre_compose_splitting_tree_mapping(self, mapping: SparseMapping[FusionTree]) -> TreePairMapping:
        """Pre-compose with a mapping that acts only on the fusion-trees."""
        res = SparseMapping[tuple[FusionTree, FusionTree]]()
        for k, self_k in self.mapping.items():
            res[k] = res_k = {}
            for (Y_j, X), self_jk in self_k.items():
                for Y_i, other_ij in mapping[Y_j].items():
                    i = (Y_i, X)
                    res_k[i] = res_k.get(i, 0) + other_ij * self_jk
        return TreePairMapping(res)

    def pre_compose_twist_instruction(self, instruction: TwistInstruction) -> TensorMapping:
        twist_mapping = SparseMapping[FusionTree]()
        if instruction.codomain:
            # because this is a splitting tree, we need to do the opposite twist to its
            # fusiontree representative, giving us one conj.
            # then, we need to conj the resulting coefficient, cancelling that conj again.
            for Y in set(Y for Y, X in self.mapping.nonzero_rows()):
                twist_mapping[Y] = Y.twist(idcs=instruction.idcs, overtwist=instruction.overtwist)
            return self.pre_compose_splitting_tree_mapping(twist_mapping)
        else:
            for X in set(X for Y, X in self.mapping.nonzero_rows()):
                twist_mapping[X] = X.twist(idcs=instruction.idcs, overtwist=instruction.overtwist)
            return self.pre_compose_fusion_tree_mapping(twist_mapping)

    def prune(self, tol: float = 1e-15) -> TreePairMapping:
        self.mapping.prune(tol=tol)

    def show(self, do_print=True, return_res=False):
        res = f'{type(self).__name__}: ( Y @ X )\n'
        indent = '    '
        for (Yi, Xi), val in self.mapping.items():
            res += f'{indent}{Yi!s}\n{indent} @ {Xi!s}\n'
            for (Yf, Xf), coeff in val.items():
                res += f'{2 * indent}{coeff:.5f}\n{2 * indent} * {Yf!s}\n{2 * indent} @ {Xf!s}\n'
            res += '\n'
        res.removesuffix('\n')
        if do_print:
            print(res)
        if return_res:
            return res

    def transform_tensor(self, data: FusionTreeData,
                         codomain: TensorProduct, domain: TensorProduct,
                         new_codomain: TensorProduct, new_domain: TensorProduct,
                         codomain_idcs: list[int], domain_idcs: list[int],
                         block_backend: BlockBackend,
                         ) -> FusionTreeData:
        # f(T)_{Jm} = sum_I f_{JI} T_{Im} = sum_I mapping[I][J] T_{Im}
        J = codomain.num_factors
        K = domain.num_factors
        N = J + K
        tree_block_axes_1 = [i if i < J else (N - 1) + (J - i) for i in codomain_idcs]
        tree_block_axes_2 = [i if i < J else (N - 1) + (J - i) for i in domain_idcs]
        inv_leg_perm = inverse_permutation([*codomain_idcs, *reversed(domain_idcs)])
        #
        dtype = data.dtype  # TODO what if blocks are real but coefficients complex???
        block_inds = []
        blocks = []
        #
        for i, j in iter_common_sorted_arrays(new_codomain.sector_decomposition,
                                              new_domain.sector_decomposition):
            coupled = new_codomain.sector_decomposition[i]
            shape = (new_codomain.block_size(i), new_domain.block_size(j))
            block = block_backend.zeros(shape, data.dtype, device=data.device)
            is_zero_block = True
            for Y, idcs1, mults1, _ in new_codomain.iter_tree_blocks([coupled]):
                for X, idcs2, mults2, _ in new_domain.iter_tree_blocks([coupled]):
                    tree_block = 0
                    is_zero_tree_block = True
                    # note: we first add all contributions to the new tree block, and do the axes
                    #       permutation only once to the result
                    for (Y_I, X_I), self_I in self.mapping.items():
                        if (Y, X) not in self_I:
                            continue
                        which_block = data.block_ind_from_coupled(Y_I.coupled, domain)
                        if which_block is None:
                            # ie old block is not set / is zero
                            continue
                        old_block = data.blocks[which_block]
                        is_zero_tree_block = False
                        i1 = codomain.tree_block_slice(Y_I)  # OPTIMIZE cache these?
                        i2 = domain.tree_block_slice(X_I)
                        tree_block += self_I[Y, X] * old_block[i1, i2]
                    if is_zero_tree_block:
                        continue
                    is_zero_block = False
                    #
                    # from the iterator, we get mults1, mults2 in the new axis order, but wee need
                    # them in the old order. OPTIMIZE can we do better than this??
                    leg_mults = [*mults1, *reversed(mults2)]
                    old_mults = [leg_mults[i] for i in inv_leg_perm]
                    #              0   1      J-1  J   J+1      J+K-1
                    # tree_block [m1, m2, ..., mJ, n1, n2, ..., nK]
                    block[idcs1, idcs2] = block_backend.permute_combined_matrix(
                        tree_block, old_mults[:J], tree_block_axes_1,
                        reversed(old_mults[J:]), tree_block_axes_2,
                    )
            if is_zero_block:
                continue
            block_inds.append([i, j])
            blocks.append(block)
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, dtype=dtype, device=data.device, is_sorted=True)


class FactorizedTreeMapping(TensorMapping):
    r"""A :class:`TensorMapping` that factorizes into maps on single trees.

    In particular, the action of the mapping on a tree pair factorizes as::

        f(Y @ X) = g(Y) @ h(X)

    and we store the component ``Y \mapsto g_{Y2, Y} Y2`` as
    ``g_{Y2, Y} = splitting_tree_mapping[Y2][Y] = <Y2 | Y>`` and similarly
    ``h_{X2, X} = fusion_tree_mapping[X2][X] = <X2 | X>`` for ``X \mapsto h_{X2, X} X2``.
    Note that ``g`` contains the coefficients in a linear combination of splitting trees,
    which are conjugated compared to the analogous linear combination of fusion trees.
    """

    def __init__(self,
                 splitting_tree_mapping: SparseMapping[FusionTree] | IdentityMapping[FusionTree],
                 fusion_tree_mapping: SparseMapping[FusionTree] | IdentityMapping[FusionTree]):
        self.splitting_tree_mapping = splitting_tree_mapping
        self.fusion_tree_mapping = fusion_tree_mapping

    @classmethod
    def from_identity(cls, codomain: TensorProduct, domain: TensorProduct,
                      block_inds: np.ndarray | None = None):
        if block_inds is None:
            block_inds = iter_common_sorted_arrays(codomain.sector_decomposition,
                                                   domain.sector_decomposition)
        splitting_trees = []
        fusion_trees = []
        for i, _ in block_inds:
            coupled = codomain.sector_decomposition[i]
            for Y, *_ in codomain.iter_tree_blocks([coupled]):
                splitting_trees.append(Y)
            for X, *_ in domain.iter_tree_blocks([coupled]):
                fusion_trees.append(X)
        splitting_tree_mapping = IdentityMapping[FusionTree](splitting_trees)
        fusion_tree_mapping = IdentityMapping[FusionTree](fusion_trees)
        return cls(splitting_tree_mapping, fusion_tree_mapping)

    def pre_compose_braid_instruction(self, instruction: BraidInstruction):
        braid_mapping = SparseMapping[FusionTree]()
        if instruction.codomain:
            # because this is a splitting tree, we need to do the opposite braid and do conj
            #   (see notes in TreePairMapping.pre_compose_braid_instruction)
            for Y in self.splitting_tree_mapping.nonzero_rows():
                braid_mapping[Y] = Y.braid(j=instruction.idx, overbraid=not instruction.overbraid,
                                           do_conj=True)
            splitting_tree_mapping = self.splitting_tree_mapping.pre_compose(braid_mapping)
            fusion_tree_mapping = self.fusion_tree_mapping
        else:
            for X in self.fusion_tree_mapping.nonzero_rows():
                braid_mapping[X] = X.braid(j=instruction.idx, overbraid=instruction.overbraid)
            splitting_tree_mapping = self.splitting_tree_mapping
            fusion_tree_mapping = self.fusion_tree_mapping.pre_compose(braid_mapping)
        return FactorizedTreeMapping(splitting_tree_mapping, fusion_tree_mapping)

    def pre_compose_bend_instruction(self, instruction):
        raise TypeError(f'{type(self).__name__} is incompatible with `{type(instruction).__name__}`.')

    def pre_compose_twist_instruction(self, instruction: TwistInstruction) -> TensorMapping:
        twist_mapping = SparseMapping[FusionTree]()
        if instruction.codomain:
            # because this is a splitting tree, we need to do the opposite twist to its
            # fusiontree representative, giving us one conj.
            # then, we need to conj the resulting coefficient, cancelling that conj again.
            for Y in self.splitting_tree_mapping.nonzero_rows():
                twist_mapping[Y] = Y.twist(idcs=instruction.idcs, overtwist=instruction.overtwist)
            splitting_tree_mapping = self.splitting_tree_mapping.pre_compose(twist_mapping)
            fusion_tree_mapping = self.fusion_tree_mapping
        else:
            for X in self.fusion_tree_mapping.nonzero_rows():
                twist_mapping[X] = X.twist(idcs=instruction.idcs, overtwist=instruction.overtwist)
            splitting_tree_mapping = self.splitting_tree_mapping
            fusion_tree_mapping = self.fusion_tree_mapping.pre_compose(twist_mapping)
        return FactorizedTreeMapping(splitting_tree_mapping, fusion_tree_mapping)

    def prune(self, tol: float = 1e-15) -> FactorizedTreeMapping:
        self.splitting_tree_mapping.prune(tol=tol)
        self.fusion_tree_mapping.prune(tol=tol)

    def show(self, do_print=True, return_res=False):
        res = f'{type(self).__name__}:\n'
        indent = '    '
        res += f'{indent}splitting_tree_mapping:\n'
        for Xi, val in self.splitting_tree_mapping.items():
            res += f'{2 * indent}{Xi!s}\n'
            for Xf, coeff in val.items():
                res += f'{3 * indent}{coeff:.5f}\n{3 * indent} * {Xf!s}\n'
        res += f'{indent}fusion_tree_mapping:\n'
        for Xi, val in self.fusion_tree_mapping.items():
            res += f'{2 * indent}{Xi!s}\n'
            for Xf, coeff in val.items():
                res += f'{3 * indent}{coeff:.5f}\n{3 * indent} * {Xf!s}\n'
        res.removesuffix('\n')
        if do_print:
            print(res)
        if return_res:
            return res

    def transform_tensor(self, data: FusionTreeData,
                         codomain: TensorProduct, domain: TensorProduct,
                         new_codomain: TensorProduct, new_domain: TensorProduct,
                         codomain_idcs: list[int], domain_idcs: list[int],
                         block_backend: BlockBackend
                         ) -> FusionTreeData:
        #
        J = codomain.num_factors
        K = domain.num_factors
        N = J + K
        #
        dtype = data.dtype  # TODO what if blocks are real but coefficients complex???
        block_inds = []
        blocks = []
        #
        for i, j in iter_common_sorted_arrays(new_codomain.sector_decomposition,
                                              new_domain.sector_decomposition):
            coupled = new_codomain.sector_decomposition[i]
            #
            which_block = data.block_ind_from_coupled(coupled, domain)
            if which_block is None:
                continue
            old_block = data.blocks[which_block]
            shape = (new_codomain.multiplicities[i], new_domain.multiplicities[j])
            #
            tmp_block = block_backend.zeros(shape, data.dtype, device=data.device)
            tmp_block, is_zero_block = self._transform_splitting_trees(
                old_block, tmp_block, coupled=coupled, codomain=codomain, new_codomain=new_codomain,
                tree_block_axes_1=codomain_idcs, block_backend=block_backend
            )
            if is_zero_block:
                continue
            #
            block = block_backend.zeros(shape, data.dtype, device=data.device)
            block, is_zero_block = self._transform_fusion_trees(
                tmp_block, block, coupled=coupled, domain=domain, new_domain=new_domain,
                tree_block_axes_2=[(N - 1) - i for i in domain_idcs], block_backend=block_backend
            )
            if is_zero_block:
                continue
            #
            block_inds.append([i, j])
            blocks.append(block)
        if len(block_inds) == 0:
            block_inds = np.zeros((0, 2), int)
        else:
            block_inds = np.array(block_inds, int)
        return FusionTreeData(block_inds, blocks, dtype=dtype, device=data.device, is_sorted=True)

    def _transform_splitting_trees(self, old_block: Block, out: Block, coupled: Sector,
                                   codomain: TensorProduct, new_codomain: TensorProduct,
                                   tree_block_axes_1: list[int], block_backend: BlockBackend
                                   ) -> tuple[Block, bool]:
        """Helper for :meth:`transform_tensor`:

        Apply :attr:`splitting_tree_mapping` to a single block.
        Write results to `out`, modifying it in-place. Usually, we pass a zero block.
        Return ``new_block, is_zero``.
        """
        if isinstance(self.splitting_tree_mapping, IdentityMapping):
            return old_block, False

        is_zero = True
        for Y2, idcs, mults, _ in new_codomain.iter_tree_blocks([coupled]):
            tree_row = 0
            is_zero_row = True
            # note: we first add all contributions to the new rows, and then do the
            #       axes permutation only once to the result.
            for Y, self_Y in self.splitting_tree_mapping.items():
                if Y2 not in self_Y:
                    continue
                is_zero_row = False
                i1 = codomain.tree_block_slice(Y)
                tree_row += self_Y[Y2] * old_block[i1, :]
            if is_zero_row:
                continue
            is_zero = False
            mults_old_order = [mults[i] for i in inverse_permutation(tree_block_axes_1)]
            out[idcs, :] = block_backend.permute_combined_idx(
                tree_row, 0, mults_old_order, tree_block_axes_1
            )

        return out, is_zero

    def _transform_fusion_trees(self, old_block: Block, out: Block, coupled: Sector,
                                domain: TensorProduct, new_domain: TensorProduct,
                                tree_block_axes_2: list[int], block_backend: BlockBackend
                                ) -> tuple[Block, bool]:
        """Helper for :meth:`transform_tensor`:

        Apply :attr:`fusion_tree_mapping` to a single block.
        Write results to `out`, modifying it in-place. Usually, we pass a zero block.
        Return ``new_block, is_zero``.
        """
        if isinstance(self.fusion_tree_mapping, IdentityMapping):
            return old_block, False
        is_zero_block = True
        for X2, idcs, mults, _ in new_domain.iter_tree_blocks([coupled]):
            tree_col = 0
            is_zero_tree_col = True
            for X, self_X in self.fusion_tree_mapping.items():
                if X2 not in self_X:
                    continue
                is_zero_tree_col = False
                i2 = domain.tree_block_slice(X)
                tree_col += self_X[X2] * old_block[:, i2]
            if is_zero_tree_col:
                continue
            is_zero_block = False
            mults_old_order = [mults[i] for i in inverse_permutation(tree_block_axes_2)]
            out[:, idcs] = block_backend.permute_combined_idx(
                tree_col, 1, mults_old_order, tree_block_axes_2
            )
        return out, is_zero_block


def _partial_trace_helper(tree: FusionTree, idcs: list[int]) -> tuple[bool, float | complex]:
    """Helper for :meth:`FusionTreeBackend.partial_trace`.

    Parameters
    ----------
    tree : FusionTree
    idcs : list of int
        Indicates which of the legs are traced: ``idcs[i]`` with ``idcs[i] + 1`` and so on.

    Returns
    -------
    contributes : bool
        If tree blocks with this tree contribute to the trace at all, i.e. if they are
        "on the diagonal" of this trace.
    b_symbol : float | complex
        The resulting B symbol.
    """
    sym = tree.symmetry
    b_symbols = 1.
    for idx in idcs:
        if not np.all(tree.uncoupled[idx] == sym.dual_sector(tree.uncoupled[idx + 1])):
            return False, 0.
        left_sec = [sym.trivial_sector, tree.uncoupled[0]]
        left_sec = left_sec[idx] if idx < 2 else tree.inner_sectors[idx - 2]
        center_sec = tree.uncoupled[0] if idx == 0 else tree.inner_sectors[idx - 1]
        right_sec = tree.inner_sectors[idx] if idx < tree.num_inner_edges else tree.coupled
        # TODO use tree.vertex_labels(idx)
        if not np.all(left_sec == right_sec):
            return False, 0.
        if idx == 0 and not np.all(tree.multiplicities[:2] == [0, 0]):
            # this must be the case if there is only one way to fuse to the trivial sector
            return False, 0.
        mu = 0 if idx == 0 else tree.multiplicities[idx - 1]
        nu = tree.multiplicities[idx]
        b_symbols *= sym.b_symbol(left_sec, tree.uncoupled[idx], center_sec)[mu, nu].conj()
        if tree.are_dual[idx]:
            b_symbols *= sym.frobenius_schur(tree.uncoupled[idx])
    return True, b_symbols


def compare_levels(level_1: int | None, level_2: int | None, has_symmetric_braid: bool) -> bool:
    """Compare levels, essentially ``level_1 > level_2``, but with edge-cases."""
    if has_symmetric_braid:
        return True
    if level_1 is None or level_2 is None:
        raise BraidChiralityUnspecifiedError('Legs that braid must have specified levels.')
    if level_1 == level_2:
        raise BraidChiralityUnspecifiedError('Legs that braid can not have the same level.')
    return level_1 > level_2
