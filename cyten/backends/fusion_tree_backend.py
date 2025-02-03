"""Implements the fusion tree backend.

.. _fusion_tree_backend__blocks:

Blocks
------
TODO elaborate about blocks, forest blocks and tree blocks
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np

from .abstract_backend import (
    TensorBackend, BlockBackend, Block, Data, DiagonalData, MaskData
)
from ..dtypes import Dtype
from ..symmetries import SectorArray, Symmetry, SymmetryError
from ..spaces import Space, ElementarySpace, TensorProduct, LegPipe
from ..trees import FusionTree, fusion_trees
from ..tools.misc import (
    inverse_permutation, iter_common_sorted_arrays, iter_common_noncommon_sorted,
    iter_common_sorted, permutation_as_swaps
)

if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask


__all__ = ['FusionTreeBackend', 'FusionTreeData', 'TreeMappingDict']


def _tree_block_iter(a: SymmetricTensor):
    sym = a.symmetry
    domain_are_dual = [sp.is_dual for sp in a.domain.factors]
    codomain_are_dual = [sp.is_dual for sp in a.codomain.factors]
    for (bi, _), block in zip(a.data.block_inds, a.data.blocks):
        coupled = a.codomain.sector_decomposition[bi]
        i1_forest = 0  # start row index of the current forest block
        i2_forest = 0  # start column index of the current forest block
        for b_sectors in a.domain.iter_uncoupled():
            tree_block_width = a.domain.tree_block_size(b_sectors)
            forest_block_width = 0
            for a_sectors in a.codomain.iter_uncoupled():
                tree_block_height = a.codomain.tree_block_size(a_sectors)
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


def _iter_sectors_mults_slices(spaces: list[Space], symmetry: Symmetry
                               ) -> Iterator[tuple[SectorArray, list[int], list[slice]]]:
    """Helper iterator over all combinations of sectors and respective mults and slices.

    Yields
    ------
    uncoupled : list of 1D array of int
        A combination ``[spaces[0].sector_decomposition[i0], spaces[1].sector_decomposition[i1], ...]``
        of uncoupled sectors
    mults : list of int
        The corresponding ``[spaces[0].multiplicities[i0], spaces[1].multiplicities[i1], ...]``.
    slices : list of slice
        The corresponding ``[slice(*spaces[0].slices[i0]), slice(*spaces[1].slices[i1]), ...]``.
    """
    if len(spaces) == 0:
        yield symmetry.empty_sector_array, [], []
        return

    if len(spaces) == 1:
        for a, m, slc in zip(spaces[0].sector_decomposition, spaces[0].multiplicities, spaces[0].slices):
            yield a[None, :], [m], [slice(*slc)]
        return

    # OPTIMIZE there is probably some itertools magic that does this better?
    # OPTIMIZE or build a grid of indices?
    for a_0, m_0, slc_0 in zip(spaces[0].sector_decomposition, spaces[0].multiplicities, spaces[0].slices):
        for a_rest, m_rest, slc_rest in _iter_sectors_mults_slices(spaces[1:], symmetry):
            yield np.concatenate([a_0[None, :], a_rest]), [m_0, *m_rest], [slice(*slc_0), *slc_rest]


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

    def block_ind_from_domain_sector_ind(self, domain_sector_ind: int) -> int | None:
        """Return `ind` such that ``block_inds[ind][1] == domain_sector_ind``"""
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

    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        raise NotImplementedError('apply_mask_to_DiagonalTensor not implemented')

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
        raise NotImplementedError('diagonal_tensor_from_full_tensor not implemented')

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
        raise NotImplementedError('diagonal_to_mask not implemented')

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
            for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(domain.factors, sym):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = domain.tree_block_size(b_sectors)
                for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(codomain.factors, sym):
                    a_dims = sym.batch_sector_dim(a_sectors)
                    tree_block_height = codomain.tree_block_size(a_sectors)
                    entries = a[(*j1, *j2)]  # [(a1,m1),...,(aJ,mJ), (b1,n1),...,(bK,nK)]
                    # reshape to [a1,m1,...,aJ,mJ, b1,n1,...,bK,nK]
                    shape = [0] * (2 * num_legs)
                    shape[::2] = [*a_dims, *b_dims]
                    shape[1::2] = m_dims + n_dims
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

    def from_random_normal(self, codomain: TensorProduct, domain: TensorProduct, sigma: float,
                           dtype: Dtype, device: str) -> Data:
        # TODO (JU) make sure we draw from the correct distribution,
        #           as defined in SymmetricTensor.from_random_normal
        raise NotImplementedError

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

    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        blocks = [self.block_backend.block_from_diagonal(block) for block in a.data.blocks]
        return FusionTreeData(a.data.block_inds, blocks, dtype=a.dtype, device=a.data.device)

    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        raise NotImplementedError('full_data_from_mask not implemented')  # TODO

    def get_device_from_data(self, a: FusionTreeData) -> str:
        return a.device

    def get_dtype_from_data(self, a: FusionTreeData) -> Dtype:
        return a.dtype

    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        raise NotImplementedError('get_element not implemented')  # TODO

    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        raise NotImplementedError('get_element_diagonal not implemented')  # TODO

    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        raise NotImplementedError('get_element_mask not implemented')  # TODO

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
        raise NotImplementedError('mask_binary_operand not implemented')

    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, TensorProduct, TensorProduct]:
        raise NotImplementedError('mask_contract_large_leg not implemented')

    def mask_contract_small_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, TensorProduct, TensorProduct]:
        raise NotImplementedError('mask_contract_small_leg not implemented')

    def mask_dagger(self, mask: Mask) -> MaskData:
        raise NotImplementedError('mask_dagger not implemented')

    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        raise NotImplementedError('mask_from_block not implemented')

    def mask_to_block(self, a: Mask) -> Block:
        raise NotImplementedError

    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> DiagonalData:
        raise NotImplementedError

    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        raise NotImplementedError('mask_transpose not implemented')

    def mask_unary_operand(self, mask: Mask, func) -> tuple[MaskData, ElementarySpace]:
        raise NotImplementedError

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
        raise NotImplementedError('outer not implemented')

    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, TensorProduct, TensorProduct]:
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
                raise SymmetryError(msg)
            # TODO do we only want to check the levels that are actually needed for the braids?
            if levels is not None:
                msg = 'inconsistent levels: there should not be a leg with a level \
                       between the levels of a pair of legs that is traced over'
                for pair in pairs:
                    for i, level in enumerate(levels):
                        if i in pair:
                            continue
                        assert (level < levels[pair[0]]) == (level < levels[pair[1]]), msg

        data, codom, dom = self.permute_legs(tensor, codomain_idcs=idcs[:num_codom_legs],
                                             domain_idcs=idcs[num_codom_legs:][::-1],
                                             levels=levels)
        # only consider coupled sectors in data that are consistent with co(domain) after tracing
        coupled = []
        for _, i in data.block_inds:
            # OPTIMIZE use sorted properties to speed this up.
            sector = dom.sector_decomposition[i]
            if new_domain.sector_decomposition_where(sector) is None:
                continue
            if new_codomain.sector_decomposition_where(sector) is None:
                continue
            coupled.append(sector)
        new_data = self.zero_data(new_codomain, new_domain, tensor.dtype, tensor.device,
                                  all_blocks=True)
        # block indices
        old_inds = [data.block_ind_from_domain_sector_ind(dom.sector_decomposition_where(c))
                    for c in coupled]
        new_inds = [new_data.block_ind_from_domain_sector_ind(new_domain.sector_decomposition_where(c))
                    for c in coupled]

        # step 2: compute new entries: iterate over all trees in the untraced
        # spaces and construct the consistent trees in the traced spaces

        def on_diagonal(tree: FusionTree, idcs: list[int]) -> tuple[bool, float | complex]:
            sym = tree.symmetry
            b_symbols = 1.
            for idx in idcs:
                if not np.all(tree.uncoupled[idx] == sym.dual_sector(tree.uncoupled[idx + 1])):
                    return False, 0.
                left_sec = [sym.trivial_sector, tree.uncoupled[0]]
                left_sec = left_sec[idx] if idx < 2 else tree.inner_sectors[idx - 2]
                center_sec = tree.uncoupled[0] if idx == 0 else tree.inner_sectors[idx - 1]
                right_sec = tree.inner_sectors[idx] if idx < tree.num_inner_edges else tree.coupled
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

        for codom_tree, codom_slc, ind in codom.iter_tree_blocks(coupled):
            on_diag, factor_codom = on_diagonal(codom_tree, codom_tree_idcs)
            if not on_diag:
                continue
            codom_shape = [codom[i].sector_multiplicity(sec)
                           for i, sec in enumerate(codom_tree.uncoupled)]
            new_codom_tree = FusionTree(
                tensor.symmetry, codom_tree.uncoupled[codom_unc_idcs], codom_tree.coupled,
                codom_tree.are_dual[codom_unc_idcs], codom_tree.inner_sectors[codom_inner_idcs],
                codom_tree.multiplicities[codom_multi_idcs]
            )
            new_codom_slc = new_codomain.tree_block_slice(new_codom_tree)
            old_ind = old_inds[ind]
            new_ind = new_inds[ind]
            for dom_tree, dom_slc, _ in dom.iter_tree_blocks([codom_tree.coupled]):
                on_diag, factor_dom = on_diagonal(dom_tree, dom_tree_idcs)
                if not on_diag:
                    continue
                dom_shape = [dom[i].sector_multiplicity(sec)
                             for i, sec in enumerate(dom_tree.uncoupled)]
                tmp_shape = tuple(codom_shape + dom_shape)
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
                     levels: list[int] | None) -> tuple[Data | None, TensorProduct, TensorProduct]:
        idcs = list(range(a.num_legs))
        if np.all(codomain_idcs == idcs[:a.num_codomain_legs]) and \
                np.all(domain_idcs == idcs[a.num_codomain_legs:][::-1]):
            return a.data, a.codomain, a.domain

        mappings, codomain, domain = TreeMappingDict.from_permute_legs(
            a=a, codomain_idcs=codomain_idcs, domain_idcs=domain_idcs, levels=levels
        )
        if mappings is None:  # levels are not given but would be needed
            return None, codomain, domain

        axes_perm = codomain_idcs + domain_idcs
        axes_perm = [i if i < a.num_codomain_legs else a.num_legs - 1 - i + a.num_codomain_legs
                     for i in axes_perm]
        data = mappings.apply_to_tensor(a, codomain, domain, axes_perm, None)
        return data, codomain, domain

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

        blocks = []
        block_inds = np.zeros((0, 2), int)
        # potential coupled sectors
        coupled_sectors = np.array([a.codomain.sector_decomposition[ind[0]] for ind in a_block_inds])
        ind_mapping = {}  # mapping between index in coupled sectors and index in blocks
        iter_space = [a.codomain, a.domain][ax_a]
        for uncoupled, slc, coupled_ind in iter_space.iter_forest_blocks(coupled_sectors):
            ind = a.domain.sector_decomposition_where(coupled_sectors[coupled_ind])
            ind_b = b.data.block_ind_from_domain_sector_ind(
                b.domain.sector_decomposition_where(uncoupled[co_domain_idx])
            )
            if ind_b is None:  # zero block
                continue

            if not ind in block_inds[:, 1]:
                ind_mapping[coupled_ind] = len(blocks)
                block_inds = np.append(
                    block_inds,
                    np.array([[a.codomain.sector_decomposition_where(a.domain.sector_decomposition[ind]), ind]]),
                    axis=0
                )
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
            forest = self.block_backend.scale_axis(forest, b_blocks[ind_b], axis=ax_a+co_domain_idx+1)
            forest = self.block_backend.reshape(forest, initial_shape)
            blocks[ind_mapping[coupled_ind]][slcs[0], slcs[1]] = forest
        return FusionTreeData(block_inds, blocks, a.dtype, a.data.device)

    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int], codomain_split: list[int],
                   domain_split: list[int], new_codomain: TensorProduct, new_domain: TensorProduct
                   ) -> Data:
        raise NotImplementedError

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
        J = len(a.codomain.factors)
        K = len(a.domain.factors)
        num_legs = J + K
        dtype = Dtype.common(a.data.dtype, a.symmetry.fusion_tensor_dtype)
        sym = a.symmetry
        # build in internal basis order first, then apply permutations in the end
        # build in codomain/domain leg order first, then permute legs in the end
        # [i1,...,iJ,j1,...,jK]
        shape = [leg.dim for leg in a.codomain.factors] + [leg.dim for leg in a.domain.factors]
        res = self.block_backend.zeros(shape, dtype)
        for bi_cod, block in zip(a.data.block_inds[:, 0], a.data.blocks):
            coupled = a.codomain.sector_decomposition[bi_cod]
            i1 = 0  # start row index of the current forest block
            i2 = 0  # start column index of the current forest block
            for b_sectors, n_dims, j2 in _iter_sectors_mults_slices(a.domain.factors, sym):
                b_dims = sym.batch_sector_dim(b_sectors)
                tree_block_width = a.domain.tree_block_size(b_sectors)
                for a_sectors, m_dims, j1 in _iter_sectors_mults_slices(a.codomain.factors, sym):
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

    def transpose(self, a: SymmetricTensor) -> tuple[Data, TensorProduct, TensorProduct]:
        codomain_idcs = list(range(a.num_codomain_legs, a.num_legs))
        domain_idcs = list(reversed(range(a.num_codomain_legs)))
        levels = list(reversed(range(a.num_legs)))
        coupled = np.array([a.domain.sector_decomposition[i[1]] for i in a.data.block_inds])

        mapping_twists = TreeMappingDict.from_topological_twists(a.codomain, coupled)
        mapping_twists = mapping_twists.add_tensorproduct(a.domain, coupled, index=1)

        mapping_permute, codomain, domain = TreeMappingDict.from_permute_legs(
            a=a, codomain_idcs=codomain_idcs, domain_idcs=domain_idcs, levels=levels
        )
        full_mapping = mapping_twists.compose(mapping_permute)

        axes_perm = list(reversed(range(a.num_legs)))
        data = full_mapping.apply_to_tensor(a, codomain, domain, axes_perm, None)
        return data, codomain, domain

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
                degeneracy_data = self.block_backend.reshape(degeneracy_data, m_dims + n_dims)
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


class TreeMappingDict(dict):
    """A description how trees need to be transformed after braiding and / or bending legs.

    This is essentially a `dict` with a few additional methods
    allowing to represent the action of b and c symbols, combining them and
    applying them to tensors, etc.

    The keys are tuples containing one or two `FusionTree`s, depending on whether
    fusion trees of codomain xor domain are transformed or of both codomain and
    domain. The values are `dict`s, where the keys correspond to the new fusion
    trees after the desired operations and the values the corresponding amplitude.

    Examples
    --------
    Two legs in the codomain are exchanged, which only affects the trees in the
    codomain. The associated `TreeMappingDict` has the form
    
    ``{(old_tree1, ) : {(new_tree1_1, ) : amplitude1_1, (new_tree1_2, ) : amplitude1_2},
    (old_tree2, ) : {...}, ...}``

    Here, the exchange process applied to `old_tree1` can be expressed as a
    superposition of the trees `new_tree1_1` and `new_tree1_2` with amplitudes
    `amplitude1_1` and `amplitude1_2`, respectively.

    Now one leg is bent up or down. This affect both codomain and domain, such that
    the associated `TreeMappingDict` has the form

    ``{(old_tree_cod1, old_tree_dom1) : {(new_tree_cod1_1, new_tree_dom1_1) : amplitude1_1,
    (new_tree_cod1_2, new_tree_dom1_2) : amplitude1_2, ...}, (old_tree_cod2, old_tree_dom2)
    : {...}, ...}``
    """
    
    def add_contribution(self, trees_i: tuple[FusionTree], trees_f: tuple[FusionTree],
                         amplitude: float | complex) -> None:
        """Add a contribution that maps `tree_i` to `tree_f`, in-place."""
        if trees_i in self:
            if trees_f in self[trees_i]:
                self[trees_i][trees_f] += amplitude
            else:
                self[trees_i][trees_f] = amplitude
        else:
            self[trees_i] = {trees_f: amplitude}

    def add_tensorproduct(self, co_domain: TensorProduct, coupled: SectorArray,
                          index: int) -> TreeMappingDict:
        """Add a product space.

        Return the `TreeMappingDict` that is obtained when adding the product space
        `co_domain` to `self`. That is, extend the `TreeMappingDict` to contain the
        fusion trees of `co_domain` without it affecting the transformation described
        by `self`, see below how the new keys may look like.

        The new `TreeMappingDict`'s keys are now tuples with one additional entry that
        corresponds to the trees in `co_domain` with coupled sector in `coupled`; the
        new product space does not affect the amplitudes in the `TreeMappingDict`.
        `index` specifies the position of the new trees within the new keys. That is,
        `index = 0` corresponds to adding `co_domain` as codomain, `index = 1` adds it
        as domain.

        This function can be used to translate `TreeMappingDict` associated c symbols to
        the level of both codomain and domain such that it can be combined with the ones
        associated with b symbols.

        For example, a `TreeMappingDict` may have the form

        ``{(old_tree1, ) : {(new_tree1_1, ) : amplitude1_1, (new_tree1_2, ) : amplitude1_2},
        (old_tree2, ) : {...}, ...}``.

        Adding a product space `prodspace` with fusion trees ``add_tree1, add_tree2, ...``
        with `index = 1` yields

        ``{(old_tree1, add_tree1) : {(new_tree1_1, add_tree1) : amplitude1_1, (new_tree1_2,
        add_tree1) : amplitude1_2}, (old_tree1, add_tree2) : {(new_tree1_1, add_tree2) :
        amplitude1_1, (new_tree1_2, add_tree2) : amplitude1_2}, ..., (old_tree2, add_tree1)
        : {...}, ...}``.
        """
        new_mapping = TreeMappingDict()
        for tree, _, _ in co_domain.iter_tree_blocks(coupled):
            for key in self:
                if not np.all(tree.coupled == key[0].coupled):
                    continue
                new_value = {self._new_key(key2, tree, index): value
                             for (key2, value) in self[key].items()}
                new_mapping[self._new_key(key, tree, index)] = new_value
        return new_mapping

    def apply_to_tensor(self, ten: SymmetricTensor, new_codomain: TensorProduct,
                        new_domain: TensorProduct, block_axes_permutation: list[int],
                        in_domain: bool | None) -> FusionTreeData:
        """Apply `self` to the tensor `ten` and return the resulting `FusionTreeData`.
        
        `new_codomain` and `new_domain` are the codomain and domain of the final
        tensor, `block_axes_permutation` gives the permutation of the axes after
        reshaping the tree blocks such that each leg corresponds to its own axis.

        If the keys in `self` contain fusion trees for both codomain and domain, all
        operations are performed on this level (as described above); if the keys only
        contain a single fusion tree, it is assumed (but not checked) that either the
        codomain XOR domain is relevant for the mapping, the other one can be ignored.
        That is, all operations are performed on the level of rows or columns of the
        blocks rather than on the level of tree blocks. This is reflected in the inputs:
        `block_axes_permutation` should then only feature the indices for the codomain
        XOR domain and `in_domain` specifies whether `self` is to be applied to the
        domain or codomain; there is no use for it in the other case.
        """
        # allow this case for more efficient treatment when only c symbols are involved
        single_fusion_tree_in_keys = False
        for key in self:
            if len(key) == 1:
                single_fusion_tree_in_keys = True
            break

        if single_fusion_tree_in_keys:
            return self._apply_single_tree_in_keys(ten, new_codomain, new_domain,
                                                   block_axes_permutation, in_domain)
        return self._apply_two_trees_in_keys(ten, new_codomain, new_domain,
                                             block_axes_permutation)

    def compose(self, dict2: TreeMappingDict) -> TreeMappingDict:
        """Compose mapping dicts.

        Return a `TreeMappingDict` corresponding to the mapping associated with first
        applying `self` and then `dict2`.
        """
        comb = TreeMappingDict()
        for key1 in self:
            comb[key1] = {}
            for key2 in self[key1]:
                for key3 in dict2[key2]:
                    if key3 in comb[key1]:
                        comb[key1][key3] += self[key1][key2] * dict2[key2][key3]
                    else:
                        comb[key1][key3] = self[key1][key2] * dict2[key2][key3]
        return comb

    @classmethod
    def compose_multiple(cls, dicts: list[TreeMappingDict]) -> TreeMappingDict:
        """Compose multiple mapping dicts.

        Return a `TreeMappingDict` corresponding to the mapping associated with applying
        all TreeMappingDict in `dicts` in the given order. It is assumed that the all keys
        have the same format (i.e., all either specified by a single or by two trees).
        """
        res = dicts[0]
        for mapping in dicts[1:]:
            res = res.compose(mapping)
        return res

    @classmethod
    def from_b_symbol(cls, codomain: TensorProduct, domain: TensorProduct, coupled: SectorArray,
                      bend_up: bool, eps: float) -> tuple[TreeMappingDict, SectorArray]:
        """From a single B move.

        Return the new coupled sectors and a `TreeMappingDict` including the details
        on how to combine the old fusion trees in `codomain` and `domain` in order to
        obtain the new ones after bending the final leg in the codomain down (`bend_up
        == False`) / domain up (`bend_up == True`). The coupled sectors `coupled`
        correspond to the coupled sectors of interest (= sectors with non-zero blocks
        of the tensor). Contributions smaller than `eps` are discarded.
        """
        symmetry = codomain.symmetry
        mapping = cls()
        new_coupled = []
        spaces = [codomain, domain]
        for tree1, _, _ in spaces[bend_up].iter_tree_blocks(coupled):
            if tree1.uncoupled.shape[0] == 1:
                new_trees_coupled = symmetry.trivial_sector
            else:
                new_trees_coupled = (tree1.inner_sectors[-1] if tree1.inner_sectors.shape[0] > 0
                                     else tree1.uncoupled[0])

            new_tree1 = FusionTree(symmetry, tree1.uncoupled[:-1], new_trees_coupled, tree1.are_dual[:-1],
                                   tree1.inner_sectors[:-1], tree1.multiplicities[:-1])

            if len(new_coupled) == 0 or not np.any( np.all(new_trees_coupled == new_coupled, axis=1) ):
                new_coupled.append(new_trees_coupled)

            b_sym = symmetry._b_symbol(new_trees_coupled, tree1.uncoupled[-1], tree1.coupled)
            if not bend_up:
                b_sym = b_sym.conj()
            if tree1.are_dual[-1]:
                b_sym = b_sym * symmetry.frobenius_schur(tree1.uncoupled[-1])
            mu = tree1.multiplicities[-1] if tree1.multiplicities.shape[0] > 0 else 0

            for tree2, _, _ in spaces[not bend_up].iter_tree_blocks([tree1.coupled]):
                if len(tree2.uncoupled) == 0:
                    new_unc = np.array([symmetry.dual_sector(tree1.uncoupled[-1])])
                    new_dual = np.array([not tree1.are_dual[-1]])
                    new_mul = np.array([], dtype=int)
                else:
                    new_unc = np.append(tree2.uncoupled, [symmetry.dual_sector(tree1.uncoupled[-1])], axis=0)
                    new_dual = np.append(tree2.are_dual, [not tree1.are_dual[-1]])
                    new_mul = np.append(tree2.multiplicities, [0]) if len(new_unc) > 2 else np.array([0])
                new_in = np.append(tree2.inner_sectors, [tree2.coupled], axis=0) if len(new_unc) > 2 else []
                new_tree2 = FusionTree(symmetry, new_unc, new_trees_coupled, new_dual, new_in, new_mul)

                for nu in range(b_sym.shape[1]):
                    if abs(b_sym[mu, nu]) < eps:
                        continue

                    # assign it only if new_tree2 has a multiplicity, i.e., more than 1 uncoupled charge
                    if len(new_tree2.uncoupled) > 1:
                        new_tree2.multiplicities[-1] = nu

                    old_trees = ([tree1, tree2][bend_up], [tree1, tree2][not bend_up])
                    new_trees = ([new_tree1, new_tree2][bend_up], [new_tree1, new_tree2][not bend_up])
                    mapping.add_contribution(old_trees, new_trees, b_sym[mu, nu])
        return mapping, np.array(new_coupled)

    @classmethod
    def from_c_symbol(cls, co_domain: TensorProduct, coupled: SectorArray, index: int,
                      overbraid: bool, in_domain: bool, eps: float) -> TreeMappingDict:
        """From a single C move.

        Return a `TreeMappingDict` including the details on how to combine the old fusion
        trees in `co_domain` in order to obtain the new ones after braiding. The braided spaces
        correspond to `index` and `index+1`;  the counting is from left to right (standard)
        in the codomain and from right to left (reverse) in the domain (if `in_domain ==
        True`). If `overbraid == True`, the space corresponding to `index` is above the one
        corresponding to `index+1`. The coupled sectors `coupled` correspond to the coupled
        sectors of interest (= sectors with non-zero blocks of the tensor). Contributions
        smaller than `eps` are discarded.
        """
        symmetry = co_domain.symmetry
        if in_domain:
            index = co_domain.num_factors - 2 - index
            overbraid = not overbraid

        mapping = cls()
        for tree, _, _ in co_domain.iter_tree_blocks(coupled):
            unc, inn, mul = tree.uncoupled, tree.inner_sectors, tree.multiplicities
            if index == 0:
                f = tree.coupled if len(inn) == 0 else inn[0]
                if overbraid:
                    factor = symmetry._r_symbol(unc[1], unc[0], f)[mul[0]]
                else:
                    factor = symmetry._r_symbol(unc[0], unc[1], f)[mul[0]].conj()
                if in_domain:
                    factor = factor.conj()

                new_tree = tree.copy(deep=True)
                new_tree.uncoupled[:2] = new_tree.uncoupled[:2][::-1]
                new_tree.are_dual[:2] = new_tree.are_dual[:2][::-1]
                mapping.add_contribution((tree, ), (new_tree, ), factor)
            else:
                left_charge = unc[0] if index == 1 else inn[index-2]
                right_charge = tree.coupled if index == inn.shape[0] else inn[index]

                for f in symmetry.fusion_outcomes(left_charge, unc[index+1]):
                    if not symmetry.can_fuse_to(f, unc[index], right_charge):
                        continue

                    new_tree = tree.copy(deep=True)
                    new_tree.inner_sectors[index-1] = f
                    new_tree.uncoupled[index:index+2] = new_tree.uncoupled[index:index+2][::-1]
                    new_tree.are_dual[index:index+2] = new_tree.are_dual[index:index+2][::-1]

                    if overbraid:
                        factors = symmetry._c_symbol(left_charge, unc[index+1], unc[index], right_charge,
                                                     f, inn[index-1])[:, :, mul[index-1], mul[index]]
                    else:
                        factors = symmetry._c_symbol(left_charge, unc[index], unc[index+1], right_charge,
                                                     inn[index-1], f)[mul[index-1], mul[index], :, :].conj()
                    if in_domain:
                        factors = factors.conj()

                    for (kap, lam), factor in np.ndenumerate(factors):
                        if abs(factor) < eps:
                            continue

                        new_tree.multiplicities[index-1] = kap
                        new_tree.multiplicities[index] = lam
                        mapping.add_contribution((tree, ), (new_tree.copy(deep=True), ), factor)
        return mapping

    @classmethod
    def from_b_or_c_symbol(cls, codomain: TensorProduct, domain: TensorProduct,
                           index: int, coupled: SectorArray, overbraid: bool | None,
                           bend_up: bool | None, backend: FusionTreeBackend
                           ) -> tuple[TreeMappingDict, TensorProduct, TensorProduct, SectorArray]:
        """Helper function.

        Essentially a wrapper for `from_b_symbol` and `from_c_symbol` that, apart
        from the corresponding `TreeMappingDict`, also returns the new codomain, new
        domain and new coupled sectors resulting from the specified operation on a
        tensor over `codomain` and `domain` with coupled charges `coupled`.

        Applies `from_b_symbol` if `index == codomain.num_spaces - 1`. Then, `bend_up`
        specifies if the final leg in the codomain is bent down or the final leg in the
        domain is bent up.

        Applies `from_c_symbol` for all other values of `index`. Then, the legs
        corresponding to `index` and `index + 1` are exchanged, with `overbraid == True`
        meaning that the leg associated with `index` is above the other leg.

        Contributions smaller than `backend.eps` are discarded.
        
        The outputs are designed such that they can be used again as input to compute
        the next step in a sequence of operations. Such sequences are then essentially
        specified by lists containing the corresponding values for `index`, `overbraid`
        and `bend_up`.

        See Also
        --------
        `from_b_symbol`, `from_c_symbol`
        """
        symmetry = codomain.symmetry
        # b symbol
        if index == codomain.num_factors - 1:
            if bend_up:
                new_domain = TensorProduct(domain.factors[:-1], symmetry)
                new_codomain = TensorProduct(codomain.factors + [domain.factors[-1].dual], symmetry)
            else:
                new_codomain = TensorProduct(codomain.factors[:-1], symmetry)
                new_domain = TensorProduct(domain.factors + [codomain.factors[-1].dual], symmetry)
            mapping, new_coupled = cls.from_b_symbol(codomain, domain, coupled,
                                                     bend_up, backend.eps)

        # c symbol
        else:
            new_coupled = coupled
            if index > codomain.num_factors - 1:
                in_domain = True
                new_codomain = codomain
                index_ = codomain.num_factors + domain.num_factors - 1 - (index + 1)
                spaces = domain.factors[:]
                spaces[index_:index_ + 2] = spaces[index_:index_ + 2][::-1]
                new_domain = TensorProduct(spaces, symmetry)
                # TODO could re-use  domain.sector_decomposition, domain.multiplicities)
                index -= codomain.num_factors
            else:
                in_domain = False
                new_domain = domain
                spaces = codomain.factors[:]
                spaces[index:index + 2] = spaces[index:index + 2][::-1]
                new_codomain = TensorProduct(spaces, symmetry)
            # TODO codomain.sector_decomposition, codomain.multiplicities)
            co_domain = [codomain, domain][in_domain]
            mapping = cls.from_c_symbol(co_domain, coupled, index, overbraid,
                                        in_domain, backend.eps)
        return mapping, new_codomain, new_domain, new_coupled

    @classmethod
    def from_permute_legs(cls, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                          levels: list[int] | None
                          ) -> tuple[TreeMappingDict, TensorProduct, TensorProduct]:
        """`permute_legs` as `TreeMappingDict`.
        
        Takes the same input as `permute_legs` in `FusionTreeBackend` and returns a
        `TreeMappingDict` corresponding to exchanging the legs of the tensor `a` as
        specified in `codomain_idcs`, `domain_idcs` and `levels`. Also returns the
        resulting codomain and domain of the resulting tensor.
        """
        # TODO special cases without bends -> can act on codomain and domain separately
        # -> do operations on the rows and columns independently rather than on the block level

        # legs that need to be bent up or down
        bend_up = sorted([i for i in codomain_idcs if i >= a.num_codomain_legs])
        bend_down = sorted([i for i in domain_idcs if i < a.num_codomain_legs])
        num_bend_up = len(bend_up)
        num_bend_down = len(bend_down)
        all_exchanges, all_bend_ups = [], []
        num_operations = []
        levels_None = (levels is None)
        if not levels_None:
            levels = levels[:]

        # exchanges such that the legs to be bent down are on the right in the codomain
        exchanges = []
        for i in range(len(bend_down)):
            for j in range(bend_down[-1 - i], a.num_codomain_legs - 1 - i):
                exchanges.append(j)
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # bend down
        all_exchanges += list(range(a.num_codomain_legs - 1, a.num_codomain_legs - 1 - num_bend_down, -1))
        all_bend_ups += [False] * num_bend_down
        num_operations.append(num_bend_down)

        # exchanges in the domain such that the legs to be bent up are on the right
        exchanges = []
        for i in range(len(bend_up)):
            for j in range(a.num_legs - bend_up[i] - 1, a.num_domain_legs + num_bend_down - 1 - i):
                exchanges.append(a.num_legs - 2 - j)
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # exchanges within the domain such that the legs agree with domain_idcs
        inter_domain_idcs = [
            i for i in range(a.num_legs-1, a.num_codomain_legs-1, -1) if not i in bend_up
        ]
        inter_domain_idcs.extend(bend_down[::-1])
        exchanges = permutation_as_swaps(inter_domain_idcs, domain_idcs)
        exchanges = [a.num_legs - 2 - i for i in exchanges]
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # bend up
        all_exchanges += list(range(a.num_codomain_legs - 1 - num_bend_down,
                                    a.num_codomain_legs - 1 - num_bend_down + num_bend_up))
        all_bend_ups += [True] * num_bend_up
        num_operations.append(num_bend_up)

        # exchanges within the codomain such that the legs agree with codomain_idcs
        inter_codomain_idcs = [i for i in range(a.num_codomain_legs) if not i in bend_down] + bend_up
        exchanges = permutation_as_swaps(inter_codomain_idcs, codomain_idcs)
        all_exchanges += exchanges
        all_bend_ups += [None] * len(exchanges)
        num_operations.append(len(exchanges))

        # c symbols are involved
        if (len(all_exchanges) - num_bend_down - num_bend_up > 0 and
                a.symmetry.braiding_style.value >= 20 and levels_None):
            # return the old codomain and domain, the new ones are not computed;
            # returning None as Data leads to a SymmetryError anyway.
            return None, a.codomain, a.domain

        codomain = a.codomain
        domain = a.domain
        coupled = np.array([domain.sector_decomposition[i[1]] for i in a.data.block_inds])
        mappings = []
        offset = [0] + list(np.cumsum(num_operations))
        for i in range(len(num_operations)):
            mappings_step = []
            for j in range(num_operations[i]):
                ind = offset[i] + j
                exchange_ind = all_exchanges[ind]
                if exchange_ind != codomain.num_factors - 1 and not levels_None:
                    overbraid = levels[exchange_ind] > levels[exchange_ind + 1]
                    levels[exchange_ind:exchange_ind + 2] = levels[exchange_ind:exchange_ind + 2][::-1]
                else:
                    overbraid = None

                mapp, codomain, domain, coupled = cls.from_b_or_c_symbol(codomain, domain, exchange_ind,
                                                                         coupled, overbraid,
                                                                         all_bend_ups[ind], a.backend)
                mappings_step.append(mapp)

            if len(mappings_step) > 0:
                mappings_step = cls.compose_multiple(mappings_step)
                if i == 0 or i == 5:
                    mappings_step = mappings_step.add_tensorproduct(domain, coupled, 1)
                elif i == 2 or i == 3:
                    mappings_step = mappings_step.add_tensorproduct(codomain, coupled, 0)
                mappings.append(mappings_step)

        mappings = cls.compose_multiple(mappings)
        return mappings, codomain, domain

    @classmethod
    def from_topological_twists(cls, co_domain: TensorProduct, coupled: SectorArray,
                                inverse: bool = False) -> TreeMappingDict:
        """From topological twists of the coupled charges.
        
        Return a `TreeMappingDict` corresponding to twisting all coupled charges
        of the fusion trees in `co_domain`, such that each tree simply acquires a
        factor equal to the topological twist of the respective coupled charge.
        If `inverse == True`, the inverse twist is applied.

        This method is useful for computing the transpose of tensors.
        """
        symmetry = co_domain.symmetry
        mapping = cls()
        for tree, _, _ in co_domain.iter_tree_blocks(coupled):
            factor = symmetry.topological_twist(tree.coupled)
            if inverse:
                factor = 1 / factor
            mapping.add_contribution((tree, ), (tree, ), factor)
        return mapping

    def _apply_single_tree_in_keys(self, ten: SymmetricTensor, new_codomain: TensorProduct,
                                   new_domain: TensorProduct, block_axes_permutation: list[int],
                                   in_domain: bool | None) -> FusionTreeData:
        backend = ten.backend.block_backend
        old_data = ten.data
        zero_blocks = [backend.zeros(backend.get_shape(block), old_data.dtype)
                       for block in old_data.blocks]
        new_data = FusionTreeData(old_data.block_inds, zero_blocks, old_data.dtype, old_data.device,
                                  True)

        if in_domain:
            iter_space, new_space = ten.domain, new_domain
        else:
            iter_space, new_space = ten.codomain, new_codomain
        old_coupled = [sec for i, sec in enumerate(ten.domain.sector_decomposition)
                       if ten.data.block_ind_from_domain_sector_ind(i) is not None]

        if in_domain:
            block_axes_permutation = [0] + [i+1 for i in block_axes_permutation]
        else:
            block_axes_permutation.append(len(block_axes_permutation))

        for tree, slc, ind in iter_space.iter_tree_blocks(old_coupled):
            modified_shape = [iter_space[i].sector_multiplicity(sec)
                              for i, sec in enumerate(tree.uncoupled)]
            if in_domain:
                block_slice = ten.data.blocks[ind][:, slc]
                modified_shape.insert(0, -1)
            else:
                block_slice = ten.data.blocks[ind][slc, :]
                modified_shape.append(-1)

            final_shape = backend.get_shape(block_slice)

            block_slice = backend.reshape(block_slice, tuple(modified_shape))
            block_slice = backend.permute_axes(block_slice, block_axes_permutation)
            block_slice = backend.reshape(block_slice, final_shape)

            contributions = self[(tree, )]
            for ((new_tree, ), amplitude) in contributions.items():
                # TODO do we want to cache the slices
                # the block indices do not change due to the way we construct new_data
                new_slc = new_space.tree_block_slice(new_tree)
                if in_domain:
                    new_data.blocks[ind][:, new_slc] += amplitude * block_slice
                else:
                    new_data.blocks[ind][new_slc, :] += amplitude * block_slice

        new_data.discard_zero_blocks(backend, ten.backend.eps)
        return new_data

    def _apply_two_trees_in_keys(self, ten: SymmetricTensor, new_codomain: TensorProduct,
                                 new_domain: TensorProduct, block_axes_permutation: list[int],
                                 ) -> FusionTreeData:
        backend = ten.backend.block_backend
        new_data = ten.backend.zero_data(new_codomain, new_domain, Dtype.complex128,
                                         device=ten.data.device, all_blocks=True)
        
        for alpha_tree, beta_tree, tree_block in _tree_block_iter(ten):
            contributions = self[(alpha_tree, beta_tree)]

            # reshape tree_block
            modified_shape = [ten.codomain[i].sector_multiplicity(sec)
                              for i, sec in enumerate(alpha_tree.uncoupled)]
            modified_shape += [ten.domain[i].sector_multiplicity(sec)
                               for i, sec in enumerate(beta_tree.uncoupled)]
            final_shape = [modified_shape[i] for i in block_axes_permutation]
            final_shape = (prod(final_shape[:new_codomain.num_factors]),
                           prod(final_shape[new_codomain.num_factors:]))

            tree_block = backend.reshape(tree_block, tuple(modified_shape))
            tree_block = backend.permute_axes(tree_block, block_axes_permutation)
            tree_block = backend.reshape(tree_block, final_shape)

            for ((new_alpha_tree, new_beta_tree), amplitude) in contributions.items():
                # TODO do we want to cache the slices and / or block_inds?
                alpha_slice = new_codomain.tree_block_slice(new_alpha_tree)
                beta_slice = new_domain.tree_block_slice(new_beta_tree)

                coupled = new_alpha_tree.coupled
                block_ind = new_domain.sector_decomposition_where(coupled)
                block_ind = new_data.block_ind_from_domain_sector_ind(block_ind)

                new_data.blocks[block_ind][alpha_slice, beta_slice] += amplitude * tree_block

        new_data.discard_zero_blocks(backend, ten.backend.eps)
        return new_data

    def _new_key(self, key, tree, index):
        newkey = list(key)
        newkey.insert(index, tree)
        return tuple(newkey)
