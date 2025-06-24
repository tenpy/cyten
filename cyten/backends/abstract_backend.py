"""TODO summary

Also contains some private utility function used by multiple backend modules.
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TypeVar, TYPE_CHECKING, Callable, Iterator
from math import prod
import numpy as np

from ..symmetries import Symmetry
from ..spaces import Space, ElementarySpace, TensorProduct, LegPipe, Leg
from ..dtypes import Dtype
from ..tools.misc import combine_constraints, to_iterable


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import SymmetricTensor, DiagonalTensor, Mask

# placeholder for a backend-specific type that holds all data of a tensor
#  (except the symmetry data stored in its legs)
Data = TypeVar('Data')
DiagonalData = TypeVar('DiagonalData')
MaskData = TypeVar('MaskData')

# placeholder for a backend-specific type that represents the blocks of symmetric tensors
Block = TypeVar('Block')


class TensorBackend(metaclass=ABCMeta):
    """Abstract base class for tensor-backends.

    A backends implements functions that act on tensors.
    We abstract two separate concepts for a backend.
    There is a block backend, that abstracts what the numerical data format (numpy array,
    torch Tensor, CUDA tensor, ...) is and a tensor-backend that abstracts how block-sparse
    structures that arise from symmetries are accounted for.

    A tensor backend has a the :attr:`block_backend` as an attribute and can call its functions
    to operate on blocks. This allows the tensor backend to be agnostic of the details of these
    blocks.
    """
    
    DataCls = None  # to be set by subclasses
    
    can_decompose_tensors = False
    """If the decompositions (SVD, QR, EIGH, ...) can operate on many-leg tensors,
    or require legs to be combined first."""

    def __init__(self, block_backend: BlockBackend):
        self.block_backend = block_backend

    def __repr__(self):
        return f'{type(self).__name__}({self.block_backend!r})'

    def __str__(self):
        return f'{type(self).__name__}({self.block_backend!r})'

    def item(self, a: SymmetricTensor | DiagonalTensor) -> float | complex:
        """Convert tensor to a python scalar.

        Assumes that tensor is a scalar (i.e. has only one entry).
        """
        return self.data_item(a.data)
    
    def test_tensor_sanity(self, a: SymmetricTensor | DiagonalTensor, is_diagonal: bool):
        """Called as part of :meth:`cyten.Tensor.test_sanity`.

        Perform sanity checks on the ``a.data``, and possibly additional backend-specific checks
        of the tensor.
        """
        # subclasses will typically call super().test_tensor_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def test_mask_sanity(self, a: Mask):
        # subclasses will typically call super().test_mask_sanity(a)
        assert isinstance(a.data, self.DataCls), str(type(a.data))

    def make_pipe(self, legs: list[Leg], is_dual: bool, in_domain: bool, pipe: LegPipe | None = None
                  ) -> LegPipe:
        """Make a pipe *of the appropriate type* for :meth:`combine_legs`.

        If `pipe` is given, try to return it if suitable.
        """
        if pipe is not None:
            assert pipe.legs == legs
            assert pipe.is_dual == is_dual
            return pipe
        return LegPipe(legs, is_dual=is_dual)

    # ABSTRACT METHODS
    
    @abstractmethod
    def act_block_diagonal_square_matrix(self, a: SymmetricTensor,
                                         block_method: Callable[[Block], Block],
                                         dtype_map: Callable[[Dtype], Dtype] | None) -> Data:
        """Apply functions like exp() and log() on a (square) block-diagonal `a`.

        Assumes the block_method returns blocks on the same device.

        Parameters
        ----------
        a : Tensor
            The tensor to act on. Can assume ``a.codomain == a.domain``.
        block_method : function
            A function with signature ``block_method(a: Block) -> Block`` acting on backend-blocks.
        dtype_map : function or None
            Specify how the result dtype depends on the input dtype. ``None`` means unchanged.
            This is needed in abelian and fusion-tree backends, in case there are 0 blocks.
        """
        ...

    @abstractmethod
    def add_trivial_leg(self, a: SymmetricTensor, legs_pos: int, add_to_domain: bool,
                        co_domain_pos: int, new_codomain: TensorProduct, new_domain: TensorProduct
                        ) -> Data:
        ...

    @abstractmethod
    def almost_equal(self, a: SymmetricTensor, b: SymmetricTensor, rtol: float, atol: float) -> bool:
        ...

    @abstractmethod
    def apply_mask_to_DiagonalTensor(self, tensor: DiagonalTensor, mask: Mask) -> DiagonalData:
        ...

    @abstractmethod
    def combine_legs(self,
                     tensor: SymmetricTensor,
                     leg_idcs_combine: list[list[int]],
                     pipes: list[LegPipe],
                     new_codomain: TensorProduct,
                     new_domain: TensorProduct,
                     ) -> Data:
        """Implementation of :func:`cyten.tensors.combine_legs`.

        Assumptions:
        
        - Legs have been permuted, such that each group of legs to be combined appears contiguously
          and either entirely in the codomain or entirely in the domain

        Parameters
        ----------
        tensor: SymmetricTensor
            The tensor to modify
        leg_idcs_combine: list of list of int
            A list of groups. Each group a list of integer leg indices, to be combined.
        pipes: list of LegPipe
            The resulting pipes. Same length and order as `leg_idcs_combine`.
            In the domain, this is the product space as it will appear in the domain, not in legs.
        new_codomain_combine:
            A list of tuples ``(positions, combined)``, where positions are all the codomain-indices
            which should be combined and ``combined`` is the resulting :class:`LegPipe`,
            i.e. ``combined == LegPipe([tensor.codomain[n] for n in positions])``
        new_domain_combine:
            Similar as `new_codomain_combine` but for the domain. Note that ``positions`` are
            domain-indices, i.e ``n = positions[i]`` refers to ``tensor.domain[n]``, *not*
            ``tensor.legs[n]`` !
        new_codomain, new_domain: TensorProduct
            The codomain and domain of the resulting tensor
        """
        ...

    @abstractmethod
    def compose(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """Assumes ``a.domain == b.codomain`` and performs contraction over those legs.
        
        Assumes there is at least one open leg, i.e. the codomain of `a` and the domain of `b` are
        not both empty. Assumes both input tensors are on the same device.
        """
        ...

    @abstractmethod
    def copy_data(self, a: SymmetricTensor | DiagonalTensor | MaskData, device: str = None
                  ) -> Data | DiagonalData | MaskData:
        """Return a copy.

        The main requirement is that future in-place operations on the output data do not affect
        the input data

        Parameters
        ----------
        a : Tensor
            The tensor to copy
        device : str, optional
            The device for the result. Per default (or if ``None``), use the same device as `a`.

        See Also
        --------
        move_to_device
        """
        ...

    @abstractmethod
    def dagger(self, a: SymmetricTensor) -> Data:
        ...

    @abstractmethod
    def data_item(self, a: Data | DiagonalData | MaskData) -> float | complex:
        """Assumes that data is a scalar (as defined in tensors.is_scalar).
        
        Return that scalar as python float or complex
        """
        ...

    @abstractmethod
    def diagonal_all(self, a: DiagonalTensor) -> bool:
        """Assumes a boolean DiagonalTensor. If all entries are True."""
        ...

    @abstractmethod
    def diagonal_any(self, a: DiagonalTensor) -> bool:
        """Assumes a boolean DiagonalTensor. If any entry is True."""
        ...

    @abstractmethod
    def diagonal_elementwise_binary(self, a: DiagonalTensor, b: DiagonalTensor, func,
                                    func_kwargs, partial_zero_is_zero: bool
                                    ) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply a function ``func(a_block: Block, b_block: Block, **kwargs) -> Block`` to all
        pairs of elements.
        Input tensors are both DiagonalTensor and have equal legs.
        ``partial_zero_is_zero=True`` promises that ``func(any_block, zero_block) == zero_block``,
        and similarly for the second argument.

        Assumes both tensors are on the same device.
        """
        ...

    @abstractmethod
    def diagonal_elementwise_unary(self, a: DiagonalTensor, func, func_kwargs, maps_zero_to_zero: bool
                                   ) -> DiagonalData:
        """Return a modified copy of the data, resulting from applying an elementwise function.

        Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
        ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
        """
        ...

    @abstractmethod
    def diagonal_from_block(self, a: Block, co_domain: TensorProduct, tol: float) -> DiagonalData:
        """The DiagonalData from a 1D block in *internal* basis order."""
        ...

    @abstractmethod
    def diagonal_from_sector_block_func(self, func, co_domain: TensorProduct) -> DiagonalData:
        """Generate diagonal data from a function.

        Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
        Assumes all generated blocks are on the same device.
        """
        ...
       
    @abstractmethod
    def diagonal_tensor_from_full_tensor(self, a: SymmetricTensor, check_offdiagonal: bool
                                         ) -> DiagonalData:
        """Get the DiagonalData corresponding to a tensor with two legs.

        Can assume that domain and codomain consist of the same single leg.
        """
        ...

    @abstractmethod
    def diagonal_tensor_trace_full(self, a: DiagonalTensor) -> float | complex:
        ...

    @abstractmethod
    def diagonal_tensor_to_block(self, a: DiagonalTensor) -> Block:
        """Forget about symmetry structure and convert to a single 1D block.
        
        This is the diagonal of the respective non-symmetric 2D tensor.
        In the *internal* basis order of the leg.
        """
        ...

    @abstractmethod
    def diagonal_to_mask(self, tens: DiagonalTensor) -> tuple[MaskData, ElementarySpace]:
        """Convert a DiagonalTensor to a Mask.

        May assume that dtype is bool.
        Returns ``mask_data, small_leg``.
        """
        ...

    @abstractmethod
    def diagonal_transpose(self, tens: DiagonalTensor) -> tuple[Space, DiagonalData]:
        """Transpose a diagonal tensor. Also return the new leg ``tens.leg.dual``"""
        ...

    @abstractmethod
    def eigh(self, a: SymmetricTensor, new_leg_dual: bool, sort: str = None
             ) -> tuple[DiagonalData, Data, ElementarySpace]:
        """Eigenvalue decomposition of a hermitian tensor

        Note that this does *not* guarantee to return the duality given by `new_leg_dual`.
        In particular, for the abelian backend, the duality is fixed.

        Parameters
        ----------
        a
        new_leg_dual : bool
            If the new leg should be dual or not.
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted *within* each charge block.
            See :func:`argsort` for details.

        Returns
        -------
        w_data
            Data for the :class:`DiagonalTensor` of eigenvalues
        v_data
            Data for the :class:`Tensor` of eigenvectors
        new_leg
            The new leg.
        """
        ...

    @abstractmethod
    def eye_data(self, co_domain: TensorProduct, dtype: Dtype, device: str) -> Data:
        """Data for :meth:``SymmetricTensor.eye``.

        The result has legs ``first_legs + [l.dual for l in reversed(firs_legs)]``.
        """
        ...

    @abstractmethod
    def from_dense_block(self, a: Block, codomain: TensorProduct, domain: TensorProduct, tol: float
                         ) -> Data:
        """Convert a dense block to the data for a symmetric tensor.

        Block is in the *internal* basis order of the respective legs and the leg order is
        ``[*codomain, *reversed(domain)]``.
        
        If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
        where ``projected`` is `a` projected to the space of symmetric tensors, raise a ``ValueError``.
        """
        ...

    @abstractmethod
    def from_dense_block_trivial_sector(self, block: Block, leg: Space) -> Data:
        """Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.

        Is given in the *internal* basis order.
        """
        ...

    @abstractmethod
    def from_random_normal(self, codomain: TensorProduct, domain: TensorProduct, sigma: float,
                           dtype: Dtype, device: str) -> Data:
        ...

    @abstractmethod
    def from_sector_block_func(self, func, codomain: TensorProduct, domain: TensorProduct) -> Data:
        """Generate tensor data from a function-

        Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
        Assumes all generated blocks are on the same device.
        """
        ...

    @abstractmethod
    def full_data_from_diagonal_tensor(self, a: DiagonalTensor) -> Data:
        ...

    @abstractmethod
    def full_data_from_mask(self, a: Mask, dtype: Dtype) -> Data:
        """May assume that the mask is a projection."""
        ...

    @abstractmethod
    def get_device_from_data(self, a: Data) -> str:
        """Extract the device from the data object"""
        ...

    @abstractmethod
    def get_dtype_from_data(self, a: Data) -> Dtype:
        ...

    @abstractmethod
    def get_element(self, a: SymmetricTensor, idcs: list[int]) -> complex | float | bool:
        """Get a single scalar element from a tensor.

        Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.
        
        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs
            - 0 <= idx < leg.dim
        """
        ...

    @abstractmethod
    def get_element_diagonal(self, a: DiagonalTensor, idx: int) -> complex | float | bool:
        """Get a single scalar element from a diagonal tensor.

        Should be equivalent to ``a.to_numpy()[idx, idx].item()`` or ``a.diagonal_as_numpy()[idx].item()``.

        Parameters
        ----------
        idx
            The index for both legs. Checks have already been performed, i.e. we may assume that
            ``0 <= idx < leg.dim``
        """
        ...

    @abstractmethod
    def get_element_mask(self, a: Mask, idcs: list[int]) -> bool:
        """Get a single scalar element from a diagonal tensor.

        Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.

        Parameters
        ----------
        idcs
            The indices. Checks have already been performed, i.e. we may assume that
            - len(idcs) == a.num_legs == 2
            - 0 <= idx < leg.dim
        """
        ...

    @abstractmethod
    def inner(self, a: SymmetricTensor, b: SymmetricTensor, do_dagger: bool) -> float | complex:
        """tensors.inner on SymmetricTensors"""
        ...

    @abstractmethod
    def inv_part_from_dense_block_single_sector(self, vector: Block, space: Space,
                                                charge_leg: ElementarySpace) -> Data:
        """Data for the invariant part used in ChargedTensor.from_dense_block_single_sector

        The vector is given in the *internal* basis order of `spaces`.
        """
        ...

    @abstractmethod
    def inv_part_to_dense_block_single_sector(self, tensor: SymmetricTensor) -> Block:
        """Inverse of inv_part_from_dense_block_single_sector

        In the *internal* basis order of `spaces`.
        """
        ...

    @abstractmethod
    def linear_combination(self, a, v: SymmetricTensor, b, w: SymmetricTensor) -> Data:
        """Form the linear combinations ``a * v + b * w``.

        Assumes `v` and `w` are on the same device.
        """
        ...

    @abstractmethod
    def lq(self, tensor: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        ...

    @abstractmethod
    def mask_binary_operand(self, mask1: Mask, mask2: Mask, func) -> tuple[MaskData, ElementarySpace]:
        """Elementwise binary function acting on two masks.

        May assume that both masks are a projection (from large to small leg)
        and that the large legs match.

        Assumes that `mask1` and `mask2` are on the same device.
        
        returns ``mask_data, new_small_leg``
        """
        ...

    @abstractmethod
    def mask_contract_large_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, TensorProduct, TensorProduct]:
        """Contraction with the large leg of a Mask.

        Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
        the large leg of the mask is contracted.
        Note that the mask may be a projection to be applied to the codomain or an inclusion
        to be contracted on the domain.
        """
        ...

    @abstractmethod
    def mask_contract_small_leg(self, tensor: SymmetricTensor, mask: Mask, leg_idx: int
                                ) -> tuple[Data, TensorProduct, TensorProduct]:
        """Contraction with the small leg of a Mask.
        
        Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
        the small leg of the mask is contracted.
        Note that the mask may be an inclusion to be applied to the codomain or a projection
        to be contracted on the domain.
        """
        ...

    @abstractmethod
    def mask_dagger(self, mask: Mask) -> MaskData:
        ...

    @abstractmethod
    def mask_from_block(self, a: Block, large_leg: Space) -> tuple[MaskData, ElementarySpace]:
        """Data for a *projection* Mask, and the resulting small leg, from a 1D block.

        a: 1D block, the Mask in *internal* basis order of `large_leg`.
        """
        ...

    @abstractmethod
    def mask_to_block(self, a: Mask) -> Block:
        """As a block of the large_leg, in *internal* basis order."""
        ...

    @abstractmethod
    def mask_to_diagonal(self, a: Mask, dtype: Dtype) -> MaskData:
        ...

    @abstractmethod
    def mask_transpose(self, tens: Mask) -> tuple[Space, Space, MaskData]:
        """Transpose a mask. Also return the new ``space_in`` and ``space_out``.

        Those spaces are the duals of the respective other in the old mask.
        """
        ...

    @abstractmethod
    def mask_unary_operand(self, mask: Mask, func) -> tuple[MaskData, ElementarySpace]:
        """Elementwise function acting on a mask.

        May assume that mask is a projection (from large to small leg).
        Returns ``mask_data, new_small_leg``
        """
        ...

    @abstractmethod
    def move_to_device(self, a: SymmetricTensor | DiagonalTensor | Mask, device: str) -> Data:
        """Move tensor to a given device.

        The result is *not* guaranteed to be a copy. In particular, if `a` already is on the
        target device, it is returned without modification.

        See Also
        --------
        copy_data
        """
        
    @abstractmethod
    def mul(self, a: float | complex, b: SymmetricTensor) -> Data:
        ...

    @abstractmethod
    def norm(self, a: SymmetricTensor | DiagonalTensor) -> float:
        """Norm of a tensor. order has already been parsed and is a number"""
        ...

    @abstractmethod
    def outer(self, a: SymmetricTensor, b: SymmetricTensor) -> Data:
        """Form the outer product, or tensor product of maps.

        Assumes that `a` and `b` are on the same device.
        """
        ...

    @abstractmethod
    def partial_trace(self, tensor: SymmetricTensor, pairs: list[tuple[int, int]],
                      levels: list[int] | None) -> tuple[Data, TensorProduct, TensorProduct]:
        """Perform an arbitrary number of traces. Pairs are converted to leg idcs.
        
        Returns ``data, codomain, domain``.
        """
        ...

    @abstractmethod
    def permute_legs(self, a: SymmetricTensor, codomain_idcs: list[int], domain_idcs: list[int],
                     levels: list[int] | None) -> tuple[Data | None, TensorProduct, TensorProduct]:
        """Permute legs on the tensors.

        codomain_idcs, domain_idcs:
            Which of the legs should end up in the (co-)domain.
            All are leg indices (``0 <= i < a.num_legs``)
        levels:
            The levels. Can assume they are unique, support comparison and are non-negative.
            ``None`` means unspecified.

        Returns
        -------
        data:
            The data for the permuted tensor, of ``None`` if `levels` are required were not specified.
        codomain, domain
            The (co-)domain of the new tensor.
        """
        ...

    @abstractmethod
    def qr(self, a: SymmetricTensor, new_co_domain: TensorProduct) -> tuple[Data, Data]:
        """Perform a QR decomposition.

        With ``a == Q @ R``
        ``Q.domain == a.domain``, ``Q.codomain == new_codomain``
        ``R.domain == new_codomain``, ``R.codomain == a.codomain``
        """
        ...

    @abstractmethod
    def reduce_DiagonalTensor(self, tensor: DiagonalTensor, block_func, func) -> float | complex:
        """Reduce a diagonal tensor to a single number.

        Used e.g. to implement ``DiagonalTensor.max``.
        ``block_func(block: Block) -> float | complex`` realizes that reduction on blocks,
        ``func(numbers: Sequence[float | complex]) -> float | complex`` for python numbers.
        """
        ...

    @abstractmethod
    def scale_axis(self, a: SymmetricTensor, b: DiagonalTensor, leg: int) -> Data:
        """Scale axis ``leg`` of ``a`` with ``b``.

        Can assume ``a.get_leg_co_domain(leg) == b.leg``.
        Assumes that `a` and `b` are on the same device.
        """
        ...

    @abstractmethod
    def split_legs(self, a: SymmetricTensor, leg_idcs: list[int], codomain_split: list[int],
                   domain_split: list[int], new_codomain: TensorProduct, new_domain: TensorProduct
                   ) -> Data:
        """Split (multiple) product space legs.

        Parameters
        ----------
        a
        leg_idcs:
            List of leg-indices, fulfilling ``0 <= i < a.num_legs``, to split.
        codomain_split, domain_split
            Contains the same information as `leg_idcs`. Which legs to split is indices for the
            (co)domain.
        new_codomain, new_domain
            The new (co-)domain, after splitting. Has same sectors and multiplicities.
        """
        ...

    @abstractmethod
    def squeeze_legs(self, a: SymmetricTensor, idcs: list[int]) -> Data:
        """Assume the legs at given indices are trivial and get rid of them"""
        ...

    @abstractmethod
    def supports_symmetry(self, symmetry: Symmetry) -> bool:
        ...

    @abstractmethod
    def svd(self, a: SymmetricTensor, new_co_domain: TensorProduct, algorithm: str | None
            ) -> tuple[Data, DiagonalData, Data]:
        ...

    @abstractmethod
    def state_tensor_product(self, state1: Block, state2: Block, pipe: LegPipe):
        """TODO clearly define what this should do in tensors.py first!

        In particular regarding basis orders.
        """
        ...

    @abstractmethod
    def to_dense_block(self, a: SymmetricTensor) -> Block:
        """Forget about symmetry structure and convert to a single block.
        
        Return a block in the *internal* basis order of the respective legs,
        with leg order ``[*codomain, *reversed(domain)]``.
        """
        ...

    @abstractmethod
    def to_dense_block_trivial_sector(self, tensor: SymmetricTensor) -> Block:
        """Single-leg tensor to the *part of* the coefficients in the trivial sector.

        In *internal* basis order.
        """
        ...

    @abstractmethod
    def to_dtype(self, a: SymmetricTensor, dtype: Dtype) -> Data:
        """Cast to given dtype. No copy if already has dtype."""
        ...

    @abstractmethod
    def trace_full(self, a: SymmetricTensor, idcs1: list[int], idcs2: list[int]) -> float | complex:
        ...

    @abstractmethod
    def transpose(self, a: SymmetricTensor) -> tuple[Data, TensorProduct, TensorProduct]:
        """Returns ``data, new_codomain, new_domain``.
        
        Note that ``new_codomain == a.domain.dual`` and ``new_domain == a.codomain.dual``.
        """
        ...

    @abstractmethod
    def truncate_singular_values(self, S: DiagonalTensor, chi_max: int | None, chi_min: int,
                                 degeneracy_tol: float, trunc_cut: float, svd_min: float
                                 ) -> tuple[MaskData, ElementarySpace, float, float]:
        """Implementation of :func:`cyten.tensors.truncate_singular_values`.

        Returns
        -------
        mask_data
            Data for the mask
        new_leg : ElementarySpace
            The new leg after truncation, i.e. the small leg of the mask
        err : float
            The truncation error ``norm(S_discard) == norm(S - S_keep)``.
        new_norm
            The norm ``norm(S_keep)`` of the approximation.
        """
        ...

    def _truncate_singular_values_selection(self, S: np.ndarray, qdims: np.ndarray | None,
                                            chi_max: int | None, chi_min: int,
                                            degeneracy_tol: float, trunc_cut: float, svd_min: float
                                            ) -> tuple[np.ndarray, float, float]:
        """Helper function for :meth:`truncate_singular_values`.

        Parameters
        ----------
        S_np : 1D numpy array of float
            A numpy array of singular values S[i]
        qdims : 1D numpy array of float
            A numpy array of the quantum dimensions. ``None`` means all qdims are one.
        chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min
            Constraints for truncation. See :func:`cyten.tensors.truncate_singular_values`.

        Returns
        -------
        mask : 1D numpy array of bool
            A boolean mask, indicating that ``S_np[mask]`` should be kept
        err : float
            The truncation error ``norm(S_discard) == norm(S - S_keep)``.
        new_norm
            The norm ``norm(S_keep)`` of the approximation.
        """
        # contributions ``err[i] = d[i] * S[i] ** 2`` to the error, if S[i] would be truncated.
        if qdims is None:
            marginal_errs = S ** 2
        else:
            marginal_errs = qdims * (S ** 2)

        # sort *ascending* by marginal errors (smallest first, should be truncated first)
        piv = np.argsort(marginal_errs)
        S = S[piv]
        # qdims = qdims[piv]  # not needed again.
        marginal_errs = marginal_errs[piv]

        # take safe logarithm, clipping small values to log(1e-100).
        # this is only used for degeneracy tol.
        logS = np.log(np.choose(S <= 1.e-100, [S, 1.e-100 * np.ones(len(S))]))

        # goal: find an index 'cut' such that we keep piv[cut:], i.e. cut between `cut-1` and `cut`.
        # build an array good, where ``good[cut] = (is `cut` an allowed choice)``.
        # we then choose the smallest good cut, i.e. we keep as many singular values as possible
        good = np.ones(len(S), dtype=bool)

        if (chi_max is not None) and (chi_max < len(S)):
            # keep at most chi_max values
            good2 = np.zeros(len(piv), dtype=np.bool_)
            good2[-chi_max:] = True
            good = combine_constraints(good, good2, "chi_max")

        if (chi_min is not None) and (chi_min > 1):
            # keep at least chi_min values
            good2 = np.ones(len(piv), dtype=np.bool_)
            good2[-chi_min + 1:] = False
            good = combine_constraints(good, good2, "chi_min")

        if (degeneracy_tol is not None) and (degeneracy_tol > 0):
            # don't cut between values (cut-1, cut) with ``log(S[cut]/S[cut-1]) < deg_tol``
            # this is equivalent to
            # ``(S[cut] - S[cut-1])/S[cut-1] < exp(deg_tol) - 1 = deg_tol + O(deg_tol^2)``
            good2 = np.empty(len(piv), np.bool_)
            good2[0] = True
            good2[1:] = np.greater_equal(logS[1:] - logS[:-1], degeneracy_tol)
            good = combine_constraints(good, good2, "degeneracy_tol")

        if (svd_min is not None):
            # keep only values S[i] >= svd_min
            good2 = np.greater_equal(S, svd_min)
            good = combine_constraints(good, good2, "svd_min")

        if (trunc_cut is not None):
            good2 = (np.cumsum(marginal_errs) > trunc_cut * trunc_cut)
            good = combine_constraints(good, good2, "trunc_cut")

        cut = np.nonzero(good)[0][0]  # smallest cut for which good[cut] is True
        err = np.sum(marginal_errs[:cut])
        new_norm = np.sum(marginal_errs[cut:])
        # build mask in the original order, before sorting
        mask = np.zeros(len(S), dtype=bool)
        np.put(mask, piv[cut:], True)
        return mask, err, new_norm

    @abstractmethod
    def zero_data(self, codomain: TensorProduct, domain: TensorProduct, dtype: Dtype, device: str,
                  all_blocks: bool = False) -> Data:
        """Data for a zero tensor.

        Parameters
        ----------
        all_blocks: bool
            Some specific backends can omit zero blocks ("sparsity").
            By default (``False``), omit them if possible.
            If ``True``, force all blocks to be created, with zero entries.
        """
        ...

    @abstractmethod
    def zero_diagonal_data(self, co_domain: TensorProduct, dtype: Dtype, device: str
                           ) -> DiagonalData:
        ...

    @abstractmethod
    def zero_mask_data(self, large_leg: Space, device: str) -> MaskData:
        ...

    def is_real(self, a: SymmetricTensor) -> bool:
        """If the Tensor is comprised of real numbers.

        Complex numbers with small or zero imaginary part still cause a `False` return.
        """
        # FusionTree backend might implement this differently.
        return a.dtype.is_real

    def save_hdf5(self, hdf5_saver, h5gr, subpath):

        hdf5_saver.save(self.block_backend, subpath + 'block_backend')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):

        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.block_backend = hdf5_loader.load(subpath + 'block_backend')


class BlockBackend(metaclass=ABCMeta):
    """Abstract base class that defines the operation on dense blocks."""
    
    svd_algorithms: list[str]  # first is default
    BlockCls = None  # to be set by subclass

    def __init__(self, default_device: str):
        self.default_device = default_device

    def __repr__(self):
        return f'{type(self).__name__}()'

    def __str__(self):
        return f'{type(self).__name__}()'

    def apply_basis_perm(self, block: Block, legs: list[Space], inv: bool = False) -> Block:
        """Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block"""
        # OPTIMIZE avoid applying permutations that we know are trivial (_basis_perm = None)
        if inv:
            perms = [leg.inverse_basis_perm for leg in legs]
        else:
            perms = [leg.basis_perm for leg in legs]
        return self.apply_leg_permutations(block, perms)

    def apply_leg_permutations(self, block: Block, perms: list[np.ndarray]) -> Block:
        """Apply permutations to every axis of a dense block"""
        return block[np.ix_(*perms)]

    @abstractmethod
    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False, device: str = None
                 ) -> Block | tuple[Block, Dtype]:
        """Convert objects to blocks.

        Should support blocks, numpy arrays, nested python containers. May support more.
        If `a` is already a block of correct dtype on the correct device, it may be returned
        un-modified.

        TODO make sure to emit warning on complex -> float!

        Returns
        -------
        block: Block
            The new block
        dtype: Dtype, optional
            The new dtype of the block. Only returned if `return_dtype`.
        device: str, optional
            The device for the block. Default behavior (if ``None``) is to leave `a` on its
            current device if it already is a block, and to use :attr:`default_device` if a new
            block needs to be created (e.g. if `a` is a list).

        See Also
        --------
        block_copy
            Guarantees an independent copy.
        """
        ...

    def as_device(self, device: str | None) -> str:
        """Convert input string to unambiguous device name.

        In particular, this should map any possible aliases to one unique name, e.g.
        for PyTorch, map ``'cuda'`` to ``'cuda:0'``.
        """
        if device is None:
            return self.default_device
        # TODO should we check if it is available here?
        return device

    @abstractmethod
    def abs_argmax(self, block: Block) -> list[int]:
        """Return the indices (one per axis) of the largest entry (by magnitude) of the block"""
        ...

    @abstractmethod
    def add_axis(self, a: Block, pos: int) -> Block:
        ...

    @abstractmethod
    def block_all(self, a) -> bool:
        """Require a boolean block. If all of its entries are True"""
        ...
        
    @abstractmethod
    def allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        ...

    @abstractmethod
    def angle(self, a: Block) -> Block:
        """The angle of a complex number such that ``a == exp(1.j * angle(a))``. Elementwise."""
        ...

    @abstractmethod
    def block_any(self, a) -> bool:
        """Require a boolean block. If any of its entries are True"""
        ...
    
    def apply_mask(self, block: Block, mask: Block, ax: int) -> Block:
        """Apply a mask (1D boolean block) to a block, slicing/projecting that axis"""
        idx = (slice(None, None, None),) * ax + (mask,)
        return block[idx]

    def argsort(self, block: Block, sort: str = None, axis: int = 0) -> Block:
        """Return the permutation that would sort a block along one axis.

        Parameters
        ----------
        block : Block
            The block to sort.
        sort : str
            Specify how the arguments should be sorted.

            ==================== =============================
            `sort`               order
            ==================== =============================
            ``'m>', 'LM'``       Largest magnitude first
            -------------------- -----------------------------
            ``'m<', 'SM'``       Smallest magnitude first
            -------------------- -----------------------------
            ``'>', 'LR', 'LA'``  Largest real part first
            -------------------- -----------------------------
            ``'<', 'SR', 'SA'``  Smallest real part first
            -------------------- -----------------------------
            ``'LI'``             Largest imaginary part first
            -------------------- -----------------------------
            ``'SI'``             Smallest imaginary part first
            ==================== =============================

        axis : int
            The axis along which to sort

        Returns
        -------
        1D block of int
            The indices that would sort the block
        """
        if sort == 'm<' or sort == 'SM':
            block = np.abs(block)
        elif sort == 'm>' or sort == 'LM':
            block = -np.abs(block)
        elif sort == '<' or sort == 'SR' or sort == 'SA':
            block = np.real(block)
        elif sort == '>' or sort == 'LR' or sort == 'LA':
            block = -np.real(block)
        elif sort == 'SI':
            block = np.imag(block)
        elif sort == 'LI':
            block = -np.imag(block)
        else:
            raise ValueError("unknown sort option " + repr(sort))
        return self._argsort(block, axis=axis)

    @abstractmethod
    def _argsort(self, block: Block, axis: int) -> Block:
        """Like :meth:`block_argsort` but can assume real valued block, and sort ascending"""
        ...

    def combine_legs(self, a: Block, leg_idcs_combine: list[list[int]], cstyles: bool | list[bool] = True) -> Block:
        """Combine each group of legs in `leg_idcs_combine` into a single leg.

        The group of legs in each entry of `leg_idcs_combine` must be contiguous.
        The legs can be combined in C style (default) or F style; the style can
        be specified for each group of legs independently.
        """
        cstyles = to_iterable(cstyles)  # single bool to list
        if len(cstyles) == 1:
            cstyles *= len(leg_idcs_combine)
        old_shape = self.get_shape(a)
        axes_perm = list(range(len(old_shape)))
        new_shape = []
        last_stop = 0
        for group, cstyle in zip(leg_idcs_combine, cstyles):
            start = group[0]
            stop = group[-1] + 1
            assert list(group) == list(range(start, stop))  # TODO rm check
            new_shape.extend(old_shape[last_stop:start])  # all leg *not* to be combined
            new_shape.append(np.prod(old_shape[start:stop]))
            if not cstyle:
                axes_perm[start:stop] = axes_perm[start:stop][::-1]
            last_stop = stop
        new_shape.extend(old_shape[last_stop:])
        return self.reshape(self.permute_axes(a, axes_perm), tuple(new_shape))

    @abstractmethod
    def conj(self, a: Block) -> Block:
        """Complex conjugate of a block"""
        ...

    @abstractmethod
    def copy_block(self, a: Block, device: str = None) -> Block:
        """Create a new, independent block with the same data

        Parameters
        ----------
        a
            The block to copy
        device
            The device for the new block. Per default, use the same device as the old block.

        See Also
        --------
        as_block
            Function to guarantee dtype and device, without forcing copies.
        """
        ...

    def dagger(self, a: Block) -> Block:
        """Permute axes to reverse order and elementwise conj."""
        num_legs = len(self.get_shape(a))
        res = self.permute_axes(a, list(reversed(range(num_legs))))
        return self.conj(res)

    @abstractmethod
    def get_dtype(self, a: Block) -> Dtype:
        ...

    @abstractmethod
    def eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        """Eigenvalue decomposition of a 2D hermitian block.

        Return a 1D block of eigenvalues and a 2D block of eigenvectors
        
        Parameters
        ----------
        block : Block
            The block to decompose
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted
        """
        ...

    @abstractmethod
    def eigvalsh(self, block: Block, sort: str = None) -> Block:
        """Eigenvalues of a 2D hermitian block.

        Return a 1D block of eigenvalues
        
        Parameters
        ----------
        block : Block
            The block to decompose
        sort : {'m>', 'm<', '>', '<'}
            How the eigenvalues are sorted
        """
        ...

    def enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(self.get_shape(block))
        shape[axis] = self.get_shape(mask)[0]
        res = self.zeros(shape, dtype=self.get_dtype(block))
        idcs = (slice(None, None, None),) * axis + (mask,)
        res[idcs] = block  # TODO mutability?
        return res

    @abstractmethod
    def exp(self, a: Block) -> Block:
        """The *elementwise* exponential.

        Not to be confused with :meth:`matrix_exp`, the *matrix* exponential.
        """
        ...

    @abstractmethod
    def block_from_diagonal(self, diag: Block) -> Block:
        """Return a 2D square block that has the 1D ``diag`` on the diagonal"""
        ...

    @abstractmethod
    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        """Convert a mask to a full block.

        Return a (N, M) of numbers (float or complex dtype) from a 1D bool-valued block shape (M,)
        where N is the number of True entries. The result is the coefficient matrix of the projection map.
        """
        ...

    @abstractmethod
    def block_from_numpy(self, a: np.ndarray, dtype: Dtype = None, device: str = None) -> Block:
        ...

    @abstractmethod
    def get_device(self, a: Block) -> str:
        ...

    @abstractmethod
    def get_diagonal(self, a: Block, check_offdiagonal: bool) -> Block:
        """Get the diagonal of a 2D block as a 1D block"""
        ...

    @abstractmethod
    def imag(self, a: Block) -> Block:
        """The imaginary part of a complex number, elementwise."""
        ...

    def inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        """Dense block version of tensors.inner.

        If do dagger, ``sum(conj(a[i1, i2, ..., iN]) * b[i1, ..., iN])``
        otherwise, ``sum(a[i1, ..., iN] * b[iN, ..., i2, i1])``.
        """
        if do_dagger:
            a = self.conj(a)
        else:
            a = self.permute_axes(a, list(reversed(range(a.ndim))))
        return self.sum_all(a * b)  # TODO or do tensordot?

    def is_real(self, a: Block) -> bool:
        """If the block is comprised of real numbers.
        
        Complex numbers with small or zero imaginary part still cause a `False` return.
        """
        return self.cyten_dtype_map[self.get_dtype(a)].is_real

    @abstractmethod
    def item(self, a: Block) -> float | complex:
        """Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex"""
        ...

    @abstractmethod
    def kron(self, a: Block, b: Block) -> Block:
        """The kronecker product.

        Parameters
        ----------
        a, b
            Two blocks with the same number of dimensions.

        Notes
        -----
        The elements are products of elements from `a` and `b`::
            kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

        where::
            kt = it * st + jt,  t = 0,...,N

        (Taken from numpy docs)
        """
        ...

    def linear_combination(self, a, v: Block, b, w: Block) -> Block:
        return a * v + b * w

    @abstractmethod
    def log(self, a: Block) -> Block:
        """The *elementwise* natural logarithm.

        Not to be confused with :meth:`matrix_log`, the *matrix* logarithm.
        """
        ...

    @abstractmethod
    def max(self, a: Block) -> float:
        ...

    @abstractmethod
    def max_abs(self, a: Block) -> float:
        ...

    @abstractmethod
    def min(self, a: Block) -> float:
        ...
        
    def mul(self, a: float | complex, b: Block) -> Block:
        return a * b

    @abstractmethod
    def norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        r"""The p-norm vector-norm of a block.

        Parameters
        ----------
        order : float
            The order :math:`p` of the norm.
            Unlike numpy, we always compute vector norms, never matrix norms.
            We only support p-norms :math:`\Vert x \Vert = \sqrt[p]{\sum_i \abs{x_i}^p}`.
        axis : int | None
            ``axis=None`` means "all axes", i.e. norm of the flattened block.
            An integer means to broadcast the norm over all other axes.
        """
        ...

    @abstractmethod
    def outer(self, a: Block, b: Block) -> Block:
        """Outer product of blocks.

        ``res[i1,...,iN,j1,...,jM] = a[i1,...,iN] * b[j1,...,jM]``
        """
        ...

    @abstractmethod
    def permute_axes(self, a: Block, permutation: list[int]) -> Block:
        ...

    @abstractmethod
    def random_normal(self, dims: list[int], dtype: Dtype, sigma: float, device: str = None
                      ) -> Block:
        ...

    @abstractmethod
    def random_uniform(self, dims: list[int], dtype: Dtype, device: str = None) -> Block:
        ...

    @abstractmethod
    def real(self, a: Block) -> Block:
        """The real part of a complex number, elementwise."""
        ...

    @abstractmethod
    def real_if_close(self, a: Block, tol: float) -> Block:
        """If a block is close to its real part, return the real part.

        Otherwise the original block. Elementwise.
        """
        ...

    @abstractmethod
    def tile(self, a: Block, repeats: int) -> Block:
        """Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat."""
        ...

    @abstractmethod
    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        ...

    @abstractmethod
    def reshape(self, a: Block, shape: tuple[int]) -> Block:
        ...

    def scale_axis(self, block: Block, factors: Block, axis: int) -> Block:
        """Multiply block with the factors (a 1D block), along a given axis.
        
        E.g. if block is 4D and ``axis==2`` with numpy-like broadcasting, this is would be
        ``block * factors[None, None, :, None]``.
        """
        idx = [None] * len(self.get_shape(block))
        idx[axis] = slice(None, None, None)
        return block * factors[tuple(idx)]

    @abstractmethod
    def get_shape(self, a: Block) -> tuple[int]:
        ...

    def split_legs(self, a: Block, idcs: list[int], dims: list[list[int]], cstyles: bool | list[bool] = True) -> Block:
        """Split legs into groups of legs with specified dimensions.

        The splitting of a leg can be in C style (default) or F style. In the
        latter case, the specified dimensions of the resulting group of legs
        *are reversed*. The style can be specified for each group of legs
        independently.
        """
        cstyles = to_iterable(cstyles)  # single bool to list
        if len(cstyles) == 1:
            cstyles *= len(idcs)
        axes_perm = []
        old_shape = self.get_shape(a)
        new_shape = []
        start = 0
        for i, i_dims, cstyle in zip(idcs, dims, cstyles):
            new_shape.extend(old_shape[start:i])
            new_shape.extend(i_dims)
            axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + i - start)))
            if cstyle:
                axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + len(i_dims))))
            else:
                axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + len(i_dims)))[::-1])
            start = i + 1
        new_shape.extend(old_shape[start:])
        axes_perm.extend(list(range(len(axes_perm), len(axes_perm) + len(old_shape) - start)))
        return self.permute_axes(self.reshape(a, tuple(new_shape)), axes_perm)

    @abstractmethod
    def sqrt(self, a: Block) -> Block:
        """The elementwise square root"""
        ...

    @abstractmethod
    def squeeze_axes(self, a: Block, idcs: list[int]) -> Block:
        ...

    @abstractmethod
    def stable_log(self, block: Block, cutoff: float) -> Block:
        """Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0."""
        ...

    @abstractmethod
    def sum(self, a: Block, ax: int) -> Block:
        """The sum over a single axis."""
        ...

    @abstractmethod
    def sum_all(self, a: Block) -> float | complex:
        """The sum of all entries of the block.
        
        If the block contains boolean values, this should return the number of ``True`` entries.
        """
        ...

    @abstractmethod
    def tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        ...

    def tensor_outer(self, a: Block, b: Block, K: int) -> Block:
        """Version of ``tensors.outer`` on blocks.

        Note the different leg order to usual outer products::

            res[i1,...,iK,j1,...,jM,i{K+1},...,iN] == a[i1,...,iN] * b[j1,...,jM]

        intended to be used with ``K == a_num_codomain_legs``.
        """
        res = self.outer(a, b)  # [i1,...,iN,j1,...,jM]
        N = len(self.get_shape(a))
        M = len(self.get_shape(b))
        return self.permute_axes(res, [*range(K), *range(N, N + M), *range(K, N)])

    @abstractmethod
    def to_dtype(self, a: Block, dtype: Dtype) -> Block:
        ...

    def to_numpy(self, a: Block, numpy_dtype=None) -> np.ndarray:
        # BlockBackends may override, if this implementation is not valid
        return np.asarray(a, dtype=numpy_dtype)

    @abstractmethod
    def trace_full(self, a: Block) -> float | complex:
        ...

    @abstractmethod
    def trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining_idcs: list[int]) -> Block:
        ...

    def eye_block(self, legs: list[int], dtype: Dtype, device: str = None) -> Data:
        """The identity matrix, reshaped to a block.

        Note the unusual leg order ``[m1,...,mJ,mJ*,...,m1*]``,
        which is chosen to match :meth:`eye_data`.

        Note also that the ``legs`` only specify the dimensions of the first half,
        namely ``m1,...,mJ``.
        """
        J = len(legs)
        eye = self.eye_matrix(prod(legs), dtype, device)
        # [M, M*] -> [m1,...,mJ,m1*,...,mJ*]
        eye = self.reshape(eye, legs * 2)
        # [m1,...,mJ,mJ*,...,m1*]
        return self.permute_axes(eye, [*range(J), *reversed(range(J, 2 * J))])

    @abstractmethod
    def eye_matrix(self, dim: int, dtype: Dtype, device: str = None) -> Block:
        """The ``dim x dim`` identity matrix"""
        ...

    @abstractmethod
    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        ...

    def get_block_mask_element(self, a: Block, large_leg_idx: int, small_leg_idx: int,
                               sum_block: int = 0) -> bool:
        """Get an element of a mask.
        
        Mask elements are `True` if the entry `a[large_leg_idx]` is the `small_leg_idx`-th `True`
        in the block.

        Parameters
        ----------
        a
            The mask block
        large_leg_idx, small_leg_idx
            The block indices
        sum_block
            Number of `True` entries in the block, i.e., ``sum_block == self.sum_all(a)``. Agrees
            with the sector multiplicity of the small leg.
            (Only important if the sector dimension is larger than 1.)
        """
        offset = (large_leg_idx // self.get_shape(a)[0]) * sum_block
        large_leg_idx = large_leg_idx % self.get_shape(a)[0]
        # if this does not work, need to override.
        if not a[large_leg_idx]:
            # if the block has a False entry, the matrix has only False in that column
            return False
        # otherwise, there is exactly one True in that column, at index sum(a[:large_leg_idx])
        return bool(small_leg_idx == offset + self.sum_all(a[:large_leg_idx]))

    @abstractmethod
    def matrix_dot(self, a: Block, b: Block) -> Block:
        """As in numpy.dot, both a and b might be matrix or vector."""
        # TODO can probably remove this? was only used in an old version of tdot.
        ...

    @abstractmethod
    def matrix_exp(self, matrix: Block) -> Block:
        ...

    @abstractmethod
    def matrix_log(self, matrix: Block) -> Block:
        ...

    def matrix_lq(self, a: Block, full: bool) -> tuple[Block, Block]:
        q, r = self.matrix_qr(self.permute_axes(a, [1, 0]), full=full)
        return self.permute_axes(r, [1, 0]), self.permute_axes(q, [1, 0])
    
    @abstractmethod
    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        """QR decomposition of a 2D block"""
        ...

    @abstractmethod
    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        """Internal version of :meth:`matrix_svd`, to be implemented by subclasses."""
        ...

    @abstractmethod
    def ones_block(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        ...

    def synchronize(self):
        """Wait for asynchronous processes (if any) to finish"""
        pass

    def test_block_sanity(self, block, expect_shape: tuple[int, ...] | None = None,
                          expect_dtype: Dtype | None = None, expect_device: str | None = None):
        assert isinstance(block, self.BlockCls), 'wrong block type'
        if expect_shape is not None:
            if self.get_shape(block) != expect_shape:
                msg = f'wrong block shape {self.get_shape(block)} != {expect_shape}'
                raise AssertionError(msg)
        if expect_dtype is not None:
            assert self.get_dtype(block) == expect_dtype, 'wrong block dtype'
        if expect_device is not None:
            assert self.get_device(block) == expect_device, 'wrong block device'

    @abstractmethod
    def zeros(self, shape: list[int], dtype: Dtype, device: str = None) -> Block:
        ...

    def save_hdf5(self, hdf5_saver, h5gr, subpath):

        hdf5_saver.save(self.BlockCls, subpath + 'BlockCls')
        hdf5_saver.save(self.svd_algorithms, subpath + 'svd_algorithms')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):

        obj = cls.__new__(cls)
        hdf5_loader.memorize_load(h5gr, obj)

        obj.BlockCls = hdf5_loader.load(subpath + 'BlockCls')
        obj.svd_algorithms = hdf5_loader.load(subpath + 'svd_algorithms')

        return obj


def conventional_leg_order(tensor_or_codomain: SymmetricTensor | TensorProduct,
                           domain: TensorProduct = None) -> Iterator[Space]:
    if domain is None:
        codomain = tensor_or_codomain.codomain
        domain = tensor_or_codomain.domain
    else:
        codomain = tensor_or_codomain
    yield from codomain.factors
    yield from reversed(domain.factors)
