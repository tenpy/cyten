#include "../py_cyten_pybind11.h"
#include "py_trampolines.hpp"

#include <cyten/backends/tensor_backend.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

void
bind_tensor_backend(py::module_& m)
{
    py::class_<TensorBackend::Data, py::smart_holder> data_cls(m, "TensorBackendData");
    data_cls.doc() = "Backend-specific payload stored on a tensor (except symmetry data on legs).";

    py::class_<TensorBackend, PyTensorBackend, py::smart_holder> tensor_backend(m,
                                                                                "TensorBackend");
    tensor_backend.doc() = R"pydoc(
Abstract base class for tensor-backends.

A backends implements functions that act on tensors.
We abstract two separate concepts for a backend.
There is a block backend, that abstracts what the numerical data format (numpy array,
torch Tensor, CUDA tensor, ...) is and a tensor-backend that abstracts how block-sparse
structures that arise from symmetries are accounted for.

A tensor backend has a the :attr:`block_backend` as an attribute and can call its functions
to operate on blocks. This allows the tensor backend to be agnostic of the details of these
blocks.
)pydoc";

    tensor_backend.def(py::init<std::shared_ptr<BlockBackend>>(), py::arg("block_backend"))
      .def_readwrite("DataCls", &TensorBackend::DataCls)
      .def_readwrite("can_decompose_tensors", &TensorBackend::can_decompose_tensors)
      .def_readwrite("block_backend", &TensorBackend::block_backend);

    tensor_backend //  methods
      .def("__repr__", &TensorBackend::__repr__)
      .def("__str__", &TensorBackend::__str__)
      .def("item",
           &TensorBackend::item,
           py::arg("a"),
           R"pydoc(
           Convert tensor to a python scalar.

           Assumes that tensor is a scalar (i.e. has only one entry).
           )pydoc")
      .def("test_tensor_sanity",
           &TensorBackend::test_tensor_sanity,
           py::arg("a"),
           py::arg("is_diagonal"),
           R"pydoc(
           Called as part of :meth:`cyten.Tensor.test_sanity`.

           Perform sanity checks on the ``a.data``, and possibly additional backend-specific checks
           of the tensor.
           )pydoc")
      .def("test_mask_sanity", &TensorBackend::test_mask_sanity, py::arg("a"))
      .def("make_pipe",
           &TensorBackend::make_pipe,
           py::arg("legs"),
           py::arg("is_dual"),
           py::arg("pipe") = py::none(),
           R"pydoc(
           Make a pipe *of the appropriate type* for :meth:`combine_legs`.

           If `pipe` is given, try to return it if suitable.
           )pydoc")
      .def("act_block_diagonal_square_matrix",
           &TensorBackend::act_block_diagonal_square_matrix,
           py::arg("a"),
           py::arg("block_method"),
           py::arg("dtype_map"),
           R"pydoc(
           Apply functions like exp() and log() on a (square) block-diagonal `a`.

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
           )pydoc")
      .def("add_trivial_leg",
           &TensorBackend::add_trivial_leg,
           py::arg("a"),
           py::arg("legs_pos"),
           py::arg("add_to_domain"),
           py::arg("co_domain_pos"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           R"pydoc(
           Add a trivial leg to a tensor.
           
           A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.
           
           Parameters
           ----------
           tens: Tensor
               The tensor to add a leg to. Since :class:`DiagonalTensor` and :class:`Mask` do not
               support adding legs, they will be converted to :class:`SymmetricTensor` first.
           legs_pos, codomain_pos, domain_pos: int
               The position of the new leg can be specified in three mutually exclusive ways.
               If the positional argument `leg_pos` is used, ``result.legs[leg_pos]`` will be the trivial
               leg. In most cases that unambiguously assigns it to either the domain or the codomain.
               If ambiguous (``if legs_pos == num_codomain_legs``), it is added to the codomain.
               Alternatively, it can be added to the codomain at ``codomain[codomain_pos]``
               or to the domain at ``domain_pos``.
               Note the implications for the ``is_dual`` argument!
               Per default, we use ``0``, i.e. add at ``legs[0]`` / ``codomain[0]``.
           label: str
               The label for the new leg.
           is_dual: bool
               If we add a dual (bra-like) or ket-like leg.
               Note that if `leg_pos` is given, we have ``result.legs[leg_pos].is_dual == is_dual``,
               but if `domain_pos` is given, we have ``result.domain[domain_pos].is_dual == is_dual``,
               which are mutually opposite.
           )pydoc")
      .def("almost_equal",
           &TensorBackend::almost_equal,
           py::arg("a"),
           py::arg("b"),
           py::arg("rtol"),
           py::arg("atol"),
           R"pydoc(
           Checks if two tensors are equal up to numerical tolerance.
           
           We compare the blocks, i.e. the free parameters of the tensors.
           The tensors count as almost equal if all block-entries, i.e. all their free parameters
           individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
           Note that this is a basis-dependent and backend-dependent notion of distance, which does
           not come from a norm in the strict mathematical sense.
           
           Parameters
           ----------
           tensor_1, tensor_2
               The tensors to compare.
           atol, rtol
               Absolute and relative tolerance, see above.
           allow_different_types: bool
               If ``True``, we convert types, e.g. via :meth:`DiagonalTensor.as_SymmetricTensor`
               to allow comparison. If ``False``, we raise on mismatching types.
           
           Notes
           -----
           Unlike numpy, our definition is symmetric under exchanging.
           
           See Also
           --------
           planar_almost_equal
               Comparison between two tensors with a possible planar permutation between them.
           )pydoc")
      .def("apply_mask_to_DiagonalTensor",
           &TensorBackend::apply_mask_to_DiagonalTensor,
           py::arg("tensor"),
           py::arg("mask"))
      .def("combine_legs",
           &TensorBackend::combine_legs,
           py::arg("tensor"),
           py::arg("leg_idcs_combine"),
           py::arg("pipes"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           R"pydoc(
           Implementation of :func:`cyten.tensors.combine_legs`.

           Assumptions:

           - Legs have been permuted, such that each group of legs to be combined appears contiguously
             and either entirely in the codomain or entirely in the domain

           Parameters
           ----------
           tensor: SymmetricTensor
               The tensor to modify
           leg_idcs_combine: list of list of int
               A list of groups. Each group a list of integer leg indices, to be combined. Must be in
               ascending order.
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
           )pydoc")
      .def("compose",
           &TensorBackend::compose,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Assumes ``a.domain == b.codomain`` and performs contraction over those legs.

           Assumes there is at least one open leg, i.e. the codomain of `a` and the domain of `b` are
           not both empty. Assumes both input tensors are on the same device.
           )pydoc")
      .def("copy_data",
           &TensorBackend::copy_data,
           py::arg("a"),
           py::arg("device") = py::none(),
           R"pydoc(
           Return a copy.

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
           )pydoc")
      .def("dagger", &TensorBackend::dagger, py::arg("a"),
      R"pydoc(
      The hermitian conjugate tensor, a.k.a the dagger of a tensor.
      
      For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
      the hermitian conjugate matrix :math:`(M^\dagger)_{i,j} = \bar{M}_{j, i}` .
      For a tensor ``A: W -> V`` the dagger is a map ``dagger(A): V -> W``.
      Graphically::
      
          |          e   d             a   b   c
          |          │   │             │   │   │
          |       ┏━━┷━━━┷━━┓         ┏┷━━━┷━━━┷┓
          |       ┃    A    ┃         ┃dagger(A)┃
          |       ┗┯━━━┯━━━┯┛         ┗━━┯━━━┯━━┛
          |        │   │   │             │   │
          |        a   b   c             e   d
      
      Where ``a, b, c, d, e`` denote the legs in to (co-)domain.
      
      Returns
      -------
      The hermitian conjugate tensor. Its legs and labels are::
      
          dagger(A).codomain == A.domain
          dagger(A).domain == A.codomain
          dagger(A).legs == [leg.dual for leg in reversed(A.legs)]
          dagger(A).labels == [_dual_leg_label(l) for l in reversed(A.labels)]
      
      Note that the resulting :attr:`Tensor.legs` only depend on the input :attr:`Tensor.legs`, not
      on their bipartition into domain and codomain.
      For labels, we toggle a duality marker, i.e. if ``A.labels == ['a', 'b', 'c', 'd*', 'e*']``,
      then ``dagger(A).labels == ['e', 'd', 'c*', 'b*','a*']``.
      )pydoc")
      .def("data_item",
           &TensorBackend::data_item,
           py::arg("a"),
           R"pydoc(
           Assumes that data is a scalar (as defined in tensors.is_scalar).

           Return that scalar as a Scalar.
           )pydoc")
      .def("diagonal_all",
           &TensorBackend::diagonal_all,
           py::arg("a"),
           R"pydoc(
           Assumes a boolean DiagonalTensor. If all entries are True.
           )pydoc")
      .def("diagonal_any",
           &TensorBackend::diagonal_any,
           py::arg("a"),
           R"pydoc(
           Assumes a boolean DiagonalTensor. If any entry is True.
           )pydoc")
      .def("diagonal_elementwise_binary",
           &TensorBackend::diagonal_elementwise_binary,
           py::arg("a"),
           py::arg("b"),
           py::arg("func"),
           py::arg("func_kwargs"),
           py::arg("partial_zero_is_zero"),
           R"pydoc(
           Return a modified copy of the data, resulting from applying an elementwise function.

           Apply a function ``func(a_block: Block, b_block: Block, **kwargs) -> Block`` to all
           pairs of elements.
           Input tensors are both DiagonalTensor and have equal legs.
           ``partial_zero_is_zero=True`` promises that ``func(any_block, zero_block) == zero_block``,
           and similarly for the second argument.

           Assumes both tensors are on the same device.
           )pydoc")
      .def("diagonal_elementwise_unary",
           &TensorBackend::diagonal_elementwise_unary,
           py::arg("a"),
           py::arg("func"),
           py::arg("func_kwargs"),
           py::arg("maps_zero_to_zero"),
           R"pydoc(
           Return a modified copy of the data, resulting from applying an elementwise function.

           Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
           ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
           )pydoc")
      .def("diagonal_from_block",
           &TensorBackend::diagonal_from_block,
           py::arg("a"),
           py::arg("co_domain"),
           py::arg("tol"),
           R"pydoc(
           The DiagonalData from a 1D block in *internal* basis order.
           )pydoc")
      .def("diagonal_from_sector_block_func",
           &TensorBackend::diagonal_from_sector_block_func,
           py::arg("func"),
           py::arg("co_domain"),
           R"pydoc(
           Generate diagonal data from a function.

           Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
           Assumes all generated blocks are on the same device.
           )pydoc")
      .def("diagonal_tensor_from_full_tensor",
           &TensorBackend::diagonal_tensor_from_full_tensor,
           py::arg("a"),
           py::arg("tol") = 1e-12,
           R"pydoc(
           Get the DiagonalData corresponding to a tensor with two legs.

           Can assume that domain and codomain consist of the same single leg.
           )pydoc")
      .def("diagonal_tensor_trace_full", &TensorBackend::diagonal_tensor_trace_full, py::arg("a"))
      .def("diagonal_tensor_to_block",
           &TensorBackend::diagonal_tensor_to_block,
           py::arg("a"),
           R"pydoc(
           Forget about symmetry structure and convert to a single 1D block.

           This is the diagonal of the respective non-symmetric 2D tensor.
           In the *internal* basis order of the leg.
           )pydoc")
      .def("diagonal_to_mask",
           &TensorBackend::diagonal_to_mask,
           py::arg("tens"),
           R"pydoc(
           Convert a DiagonalTensor to a Mask.

           May assume that dtype is bool.
           Returns ``mask_data, small_leg``.
           )pydoc")
      .def("diagonal_transpose",
           &TensorBackend::diagonal_transpose,
           py::arg("tens"),
           R"pydoc(
           Transpose a diagonal tensor. Also return the new leg ``tens.leg.dual``
           )pydoc")
      .def("eigh",
           &TensorBackend::eigh,
           py::arg("a"),
           py::arg("new_leg_dual"),
           py::arg("sort") = py::none(),
           R"pydoc(
           Eigenvalue decomposition of a hermitian tensor

           Note that this does *not* guarantee to return the duality given by `new_leg_dual`.
           In particular, for the abelian backend, the duality is fixed.

           Parameters
           ----------
           a
               The input tensor. Assumed to be hermitian without checking!
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
           )pydoc")
      .def("eye_data",
           &TensorBackend::eye_data,
           py::arg("co_domain"),
           py::arg("dtype"),
           py::arg("device"),
           R"pydoc(
           Data for :meth:``SymmetricTensor.eye``.

           The result has legs ``first_legs + [l.dual for l in reversed(firs_legs)]``.
           )pydoc")
      .def("from_dense_block",
           &TensorBackend::from_dense_block,
           py::arg("a"),
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("tol"),
           R"pydoc(
           Convert a dense block to the data for a symmetric tensor.

           Block is in the *internal* basis order of the respective legs and the leg order is
           ``[*codomain, *reversed(domain)]``.

           If the block is not symmetric, measured by ``allclose(a, projected, atol, rtol)``,
           where ``projected`` is `a` projected to the space of symmetric tensors, raise a ``ValueError``.
           )pydoc")
      .def("from_dense_block_trivial_sector",
           &TensorBackend::from_dense_block_trivial_sector,
           py::arg("block"),
           py::arg("leg"),
           R"pydoc(
           Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.

           Is given in the *internal* basis order.
           )pydoc")
      .def("from_grid",
           &TensorBackend::from_grid,
           py::arg("grid"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           py::arg("left_mult_slices"),
           py::arg("right_mult_slices"),
           py::arg("dtype"),
           py::arg("device"),
           R"pydoc(
           Data from a grid of tensors.

           Parameters
           ----------
           grid: list[list[SymmetricTensor | None]]
               Contains the tensors from which a single tensor is constructed. `None` entries are
               interpreted as tensors with all blocks equal to zero.
           new_codomain: TensorProduct
               Codomain of the resulting tensor after stacking the tensors in the grid.
           new_domain: TensorProduct
               Domain of the resulting tensor after stacking the tensors in the grid.
           left_mult_slices: list[list[int]]
               Multiplicity slices for each sector for the stacking in the codomain. That is,
               ``slice(left_mult_slices[sector_idx][i], left_mult_slices[sector_idx][i + 1])`` is the
               slice that is contributed from the tensors in the `i`th column to the sector
               ``new_codomain[0].sector_decomposition[sector_idx]`` of the leg ``new_codomain[0]``.
           right_mult_slices: list[list[int]]
               Multiplicity slices for each sector for the stacking in the domain. That is,
               ``slice(right_mult_slices[sector_idx][i], right_mult_slices[sector_idx][i + 1])`` is
               the slice that is contributed from the tensors in the `i`th row to the sector
               ``new_domain[-1].sector_decomposition[sector_idx]`` of the leg ``new_domain[-1]``.
           dtype: Dtype
               The new dtype of the block.
           device: str
               The device for the block.
           )pydoc")
      .def("from_random_normal",
           &TensorBackend::from_random_normal,
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("sigma"),
           py::arg("dtype"),
           py::arg("device"))
      .def("from_sector_block_func",
           &TensorBackend::from_sector_block_func,
           py::arg("func"),
           py::arg("codomain"),
           py::arg("domain"),
           R"pydoc(
           Generate tensor data from a function-

           Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
           Assumes all generated blocks are on the same device.
           )pydoc")
      .def("from_tree_pairs",
           &TensorBackend::from_tree_pairs,
           py::arg("trees"),
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("dtype"),
           py::arg("device"),
           R"pydoc(
           Compute the data for :meth:`SymmetricTensor.from_tree_pairs`.
           )pydoc")
      .def("full_data_from_diagonal_tensor",
           &TensorBackend::full_data_from_diagonal_tensor,
           py::arg("a"))
      .def("full_data_from_mask",
           &TensorBackend::full_data_from_mask,
           py::arg("a"),
           py::arg("dtype"),
           R"pydoc(
           May assume that the mask is a projection.
           )pydoc")
      .def("get_device_from_data",
           &TensorBackend::get_device_from_data,
           py::arg("a"),
           R"pydoc(
           Extract the device from the data object
           )pydoc")
      .def("get_dtype_from_data", &TensorBackend::get_dtype_from_data, py::arg("a"))
      .def("get_element",
           &TensorBackend::get_element,
           py::arg("a"),
           py::arg("idcs"),
           R"pydoc(
           Get a single scalar element from a tensor.

           Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.

           Parameters
           ----------
           idcs
               The indices. Checks have already been performed, i.e. we may assume that
               - len(idcs) == a.num_legs
               - 0 <= idx < leg.dim
           )pydoc")
      .def("get_element_diagonal",
           &TensorBackend::get_element_diagonal,
           py::arg("a"),
           py::arg("idx"),
           R"pydoc(
           Get a single scalar element from a diagonal tensor.

           Should be equivalent to ``a.to_numpy()[idx, idx].item()`` or ``a.diagonal_as_numpy()[idx].item()``.

           Parameters
           ----------
           idx
               The index for both legs. Checks have already been performed, i.e. we may assume that
               ``0 <= idx < leg.dim``
           )pydoc")
      .def("get_element_mask",
           &TensorBackend::get_element_mask,
           py::arg("a"),
           py::arg("idcs"),
           R"pydoc(
           Get a single scalar element from a diagonal tensor.

           Should be equivalent to ``a.to_numpy()[tuple(idcs)].item()``.

           Parameters
           ----------
           idcs
               The indices. Checks have already been performed, i.e. we may assume that
               - len(idcs) == a.num_legs == 2
               - 0 <= idx < leg.dim
           )pydoc")
      .def("inner",
           &TensorBackend::inner,
           py::arg("a"),
           py::arg("b"),
           py::arg("do_dagger"),
           R"pydoc(
           tensors.inner on SymmetricTensors
           )pydoc")
      .def("inv_part_from_dense_block_single_sector",
           &TensorBackend::inv_part_from_dense_block_single_sector,
           py::arg("vector"),
           py::arg("space"),
           py::arg("charge_leg"),
           R"pydoc(
           Data for the invariant part used in ChargedTensor.from_dense_block_single_sector

           The vector is given in the *internal* basis order of `spaces`.
           )pydoc")
      .def("inv_part_to_dense_block_single_sector",
           &TensorBackend::inv_part_to_dense_block_single_sector,
           py::arg("tensor"),
           R"pydoc(
           Inverse of inv_part_from_dense_block_single_sector

           In the *internal* basis order of `spaces`.
           )pydoc")
      .def("linear_combination",
           &TensorBackend::linear_combination,
           py::arg("a"),
           py::arg("v"),
           py::arg("b"),
           py::arg("w"),
           R"pydoc(
           Form the linear combinations ``a * v + b * w``.

           Assumes `v` and `w` are on the same device.
           )pydoc")
      .def("lq", &TensorBackend::lq, py::arg("tensor"), py::arg("new_co_domain"),
      R"pydoc(
      The LQ decomposition of a tensor.
      
      A :ref:`tensor decomposition <decompositions>` ``tensor ~ L @ Q`` with the following
      properties:
      
      - ``L`` has a lower triangular structure *in the coupled basis*.
      - ``Q`` is an isometry: ``dagger(Q) @ Q ~ eye``.
      
      Graphically::
      
          |                                 │   │   │   │
          |                                ┏┷━━━┷━━━┷━━━┷┓
          |        │   │   │   │           ┃      Q      ┃
          |       ┏┷━━━┷━━━┷━━━┷┓          ┗━━━━━━┯━━━━━━┛
          |       ┃   tensor    ┃    ==           │
          |       ┗━━┯━━━┯━━━┯━━┛          ┏━━━━━━┷━━━━━━┓
          |          │   │   │             ┃      L      ┃
          |                                ┗━━┯━━━┯━━━┯━━┛
          |                                   │   │   │
      
      We always compute the "reduced", a.k.a. "economic" version.
      To group the legs differently, use :func:`permute_legs` or `combine_to_matrix` first.
      
      Parameters
      ----------
      tensor: :class:`Tensor`
          The tensor to decompose.
      new_labels: (list of) str
          Labels for the new legs. Either two legs ``[a, b]`` s.t. ``L.labels[-1] == a``
          and ``Q.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
      new_leg_dual: bool
          If the new leg should be a ket space (``False``) or bra space (``True``).
      charge_leg_top: bool
          Fixes whether the charge leg of a decomposed :class:`ChargedTensor` should end up in the
          top tensor ``Q`` (``True``) or the bottom tensor ``L`` (``False``). The corresponding
          tensor is then also a `ChargedTensor`. Is ignored if the input tensor is not a
          `ChargedTensor`.
      )pydoc")
      .def("mask_binary_operand",
           &TensorBackend::mask_binary_operand,
           py::arg("mask1"),
           py::arg("mask2"),
           py::arg("func"),
           R"pydoc(
           Elementwise binary function acting on two masks.

           May assume that both masks are a projection (from large to small leg)
           and that the large legs match.

           Assumes that `mask1` and `mask2` are on the same device.

           returns ``mask_data, new_small_leg``
           )pydoc")
      .def("mask_contract_large_leg",
           &TensorBackend::mask_contract_large_leg,
           py::arg("tensor"),
           py::arg("mask"),
           py::arg("leg_idx"),
           R"pydoc(
           Contraction with the large leg of a Mask.

           Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
           the large leg of the mask is contracted.
           Note that the mask may be a projection to be applied to the codomain or an inclusion
           to be contracted on the domain.
           )pydoc")
      .def("mask_contract_small_leg",
           &TensorBackend::mask_contract_small_leg,
           py::arg("tensor"),
           py::arg("mask"),
           py::arg("leg_idx"),
           R"pydoc(
           Contraction with the small leg of a Mask.

           Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
           the small leg of the mask is contracted.
           Note that the mask may be an inclusion to be applied to the codomain or a projection
           to be contracted on the domain.
           )pydoc")
      .def("mask_dagger", &TensorBackend::mask_dagger, py::arg("mask"))
      .def("mask_from_block",
           &TensorBackend::mask_from_block,
           py::arg("a"),
           py::arg("large_leg"),
           R"pydoc(
           Data for a *projection* Mask, and the resulting small leg, from a 1D block.

           a: 1D block, the Mask in *internal* basis order of `large_leg`.
           )pydoc")
      .def("mask_to_block",
           &TensorBackend::mask_to_block,
           py::arg("a"),
           R"pydoc(
           As a block of the large_leg, in *internal* basis order.
           )pydoc")
      .def("mask_to_diagonal", &TensorBackend::mask_to_diagonal, py::arg("a"), py::arg("dtype"))
      .def("mask_transpose",
           &TensorBackend::mask_transpose,
           py::arg("tens"),
           R"pydoc(
           Transpose a mask. Also return the new ``space_in`` and ``space_out``.

           Those spaces are the duals of the respective other in the old mask.
           )pydoc")
      .def("mask_unary_operand",
           &TensorBackend::mask_unary_operand,
           py::arg("mask"),
           py::arg("func"),
           R"pydoc(
           Elementwise function acting on a mask.

           May assume that mask is a projection (from large to small leg).
           Returns ``mask_data, new_small_leg``
           )pydoc")
      .def("move_to_device",
           &TensorBackend::move_to_device,
           py::arg("a"),
           py::arg("device"),
           R"pydoc(
           Move tensor to a given device.

           The result is *not* guaranteed to be a copy. In particular, if `a` already is on the
           target device, it is returned without modification.

           See Also
           --------
           copy_data
           )pydoc")
      .def("mul", &TensorBackend::mul, py::arg("a"), py::arg("b"))
      .def("norm",
           &TensorBackend::norm,
           py::arg("a"),
           R"pydoc(
           Norm of a tensor. order has already been parsed and is a number
           )pydoc")
      .def("outer",
           &TensorBackend::outer,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Form the outer product, or tensor product of maps.

           Assumes that `a` and `b` are on the same device.
           )pydoc")
      .def("partial_compose",
           &TensorBackend::partial_compose,
           py::arg("a"),
           py::arg("b"),
           py::arg("a_first_leg"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           R"pydoc(
           Contract the codomain (domain) of `b` with the a part of the domain (codomain) of `a`.

           Assumes that there is at least one open leg in the domain (codomain) of the resulting
           tensor. Assumes both input tensors are on the same device.
           )pydoc")
      .def("partial_trace",
           &TensorBackend::partial_trace,
           py::arg("tensor"),
           py::arg("pairs"),
           py::arg("levels"),
           R"pydoc(
           Perform an arbitrary number of traces. Pairs are converted to leg idcs.

           Returns ``data, codomain, domain``.
           )pydoc")
      .def("permute_legs",
           &TensorBackend::permute_legs,
           py::arg("a"),
           py::arg("codomain_idcs"),
           py::arg("domain_idcs"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           py::arg("mixes_codomain_domain"),
           py::arg("levels"),
           py::arg("bend_right"),
           R"pydoc(
           Permute legs on the tensors.

           Parameters
           ----------
           a : SymmetricTensor
               The tensor to act on.
           codomain_idcs, domain_idcs:
               Which of the legs should end up in the (co-)domain.
               All are leg indices (``0 <= i < a.num_legs``).
           new_codomain, new_domain : TensorProduct
               The (co)domain of the result.
           mixes_codomain_domain : bool
               If any leg moves from the codomain to the domain or vv during the permutation.
           levels:
               The levels. Must support comparison with ``<`` or be ``None``, meaning unspecified.
           bend_right:
               For each leg, whether it bends to the left or right of the tensor.
               ``None`` is allowed as a placeholder, only if that leg does not bend at all.
               Note that non-bending legs do not necessarily have a ``None`` entry, however.

           Returns
           -------
           data:
               The data for the permuted tensor, or ``None`` if `levels` are required but were not
               specified.
           codomain, domain
               The (co-)domain of the new tensor.
           )pydoc")
      .def("qr",
           &TensorBackend::qr,
           py::arg("a"),
           py::arg("new_co_domain"),
           R"pydoc(
           Perform a QR decomposition.

           With ``a == Q @ R``
           ``Q.domain == a.domain``, ``Q.codomain == new_codomain``
           ``R.domain == new_codomain``, ``R.codomain == a.codomain``
           )pydoc")
      .def("reduce_DiagonalTensor",
           &TensorBackend::reduce_DiagonalTensor,
           py::arg("tensor"),
           py::arg("block_func"),
           py::arg("func"),
           R"pydoc(
           Reduce a diagonal tensor to a single number.

           Used e.g. to implement ``DiagonalTensor.max``.
           ``block_func(block: Block) -> Scalar`` realizes that reduction on blocks,
           ``func(numbers: Sequence[Scalar]) -> Scalar`` for numbers.
           )pydoc")
      .def("scale_axis",
           &TensorBackend::scale_axis,
           py::arg("a"),
           py::arg("b"),
           py::arg("leg"),
           R"pydoc(
           Scale axis ``leg`` of ``a`` with ``b``.

           Can assume ``a.get_leg_co_domain(leg) == b.leg``.
           Assumes that `a` and `b` are on the same device.
           )pydoc")
      .def("split_legs",
           &TensorBackend::split_legs,
           py::arg("a"),
           py::arg("leg_idcs"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           R"pydoc(
           Split (multiple) product space legs.

           Parameters
           ----------
           a
               The tensor to split legs on.
           leg_idcs:
               List of leg-indices, fulfilling ``0 <= i < a.num_legs``, to split. Must be in
               ascending order.
           new_codomain, new_domain
               The new (co-)domain, after splitting. Has same sectors and multiplicities.
           )pydoc")
      .def("squeeze_legs",
           &TensorBackend::squeeze_legs,
           py::arg("a"),
           py::arg("idcs"),
           R"pydoc(
           Assume the legs at given indices are trivial and get rid of them
           )pydoc")
      .def("supports_symmetry", &TensorBackend::supports_symmetry, py::arg("symmetry"))
      .def(
        "svd", &TensorBackend::svd, py::arg("a"), py::arg("new_co_domain"), py::arg("algorithm"),
        R"pydoc(
        The singular value decomposition (SVD) of a tensor.
        
        A :ref:`tensor decomposition <decompositions>` ``tensor ~ U @ S @ Vh`` with the following
        properties:
        
        - ``Vh`` and ``U`` are isometries: ``dagger(U) @ U ~ eye ~ Vh @ dagger(Vh)``.
        - ``S`` is a :class:`DiagonalTensor` with real, non-negative entries.
        - If `tensor` is a matrix (i.e. if it has exactly one leg each in domain and codomain), it
          reproduces the usual matrix SVD.
        
        .. note ::
            The basis for the newly generated leg is chosen arbitrarily, and in particular, unlike,
            e.g., :func:`numpy.linalg.svd` it is not guaranteed that ``S.diag_numpy`` is sorted.
        
        Graphically::
        
            |                                 │   │   │   │
            |                                ┏┷━━━┷━━━┷━━━┷┓
            |                                ┃      Vh     ┃
            |        │   │   │   │           ┗━━━━━━┯━━━━━━┛
            |       ┏┷━━━┷━━━┷━━━┷┓               ┏━┷━┓
            |       ┃   tensor    ┃    ==         ┃ S ┃
            |       ┗━━┯━━━┯━━━┯━━┛               ┗━┯━┛
            |          │   │   │             ┏━━━━━━┷━━━━━━┓
            |                                ┃      U      ┃
            |                                ┗━━┯━━━┯━━━┯━━┛
            |                                   │   │   │
        
        We always compute the "reduced", a.k.a. "economic" version of SVD, where the isometries are
        (in general) not full unitaries.
        
        To group the legs differently, use :func:`permute_legs` or `combine_to_matrix` first.
        
        Parameters
        ----------
        tensor: :class:`Tensor`
            The tensor to decompose.
        new_labels: (list of) str, optional
            The labels for the new legs can be specified in the following three ways;
            Four labels ``[a, b, c, d]`` result in ``U.labels[-1] == a``, ``S.labels == [b, c]`` and
            ``Vh.labels[0] == d``.
            Two labels ``[a, b]`` are equivalent to ``[a, b, a, b]``.
            A single label ``a`` is equivalent to ``[a, a*, a, a*]``.
            The new legs are unlabelled by default.
        new_leg_dual: bool
            If the new leg should be a ket space (``False``) or bra space (``True``).
        charge_leg_top: bool
            Fixes whether the charge leg of a decomposed :class:`ChargedTensor` should end up in the
            top tensor ``Vh`` (``True``) or the bottom tensor ``U`` (``False``). The corresponding
            tensor is then also a `ChargedTensor`. Is ignored if the input tensor is not a
            `ChargedTensor`.
        algorithm: str, optional
            The algorithm (a.k.a. "driver") for the block-wise svd. Choices are backend-specific.
            See :meth:`~cyten.block_backends.BlockBackend.possible_svd_algorithms`.
        
        Returns
        -------
        U: SymmetricTensor | ChargedTensor
        S: DiagonalTensor
        Vh: SymmetricTensor | ChargedTensor
        )pydoc")
      .def("state_tensor_product",
           &TensorBackend::state_tensor_product,
           py::arg("state1"),
           py::arg("state2"),
           py::arg("pipe"),
           R"pydoc(
           TODO clearly define what this should do in tensors.py first!

           In particular regarding basis orders.
           )pydoc")
      .def("to_block_backend",
           &TensorBackend::to_block_backend,
           py::arg("data"),
           py::arg("block_backend"),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none())
      .def("to_dense_block",
           &TensorBackend::to_dense_block,
           py::arg("a"),
           R"pydoc(
           Forget about symmetry structure and convert to a single block.

           Return a block in the *internal* basis order of the respective legs,
           with leg order ``[*codomain, *reversed(domain)]``.
           )pydoc")
      .def("to_dense_block_trivial_sector",
           &TensorBackend::to_dense_block_trivial_sector,
           py::arg("tensor"),
           R"pydoc(
           Single-leg tensor to the *part of* the coefficients in the trivial sector.

           In *internal* basis order.
           )pydoc")
      .def("to_dtype",
           &TensorBackend::to_dtype,
           py::arg("a"),
           py::arg("dtype"),
           R"pydoc(
           Cast to given dtype. No copy if already has dtype.
           )pydoc")
      .def("trace_full",
           &TensorBackend::trace_full,
           py::arg("a"),
           py::arg("idcs1") = std::vector<int64>{},
           py::arg("idcs2") = std::vector<int64>{})
      .def("truncate_singular_values",
           &TensorBackend::truncate_singular_values,
           py::arg("S"),
           py::arg("chi_max"),
           py::arg("chi_min"),
           py::arg("degeneracy_tol"),
           py::arg("trunc_cut"),
           py::arg("svd_min"),
           py::arg("minimize_error") = true,
           R"pydoc(
           Implementation of :func:`cyten.tensors.truncate_singular_values`.

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
           )pydoc")
      .def("_truncate_singular_values_selection",
           &TensorBackend::_truncate_singular_values_selection,
           py::arg("S"),
           py::arg("qdims"),
           py::arg("chi_max"),
           py::arg("chi_min"),
           py::arg("degeneracy_tol"),
           py::arg("trunc_cut"),
           py::arg("svd_min"),
           py::arg("minimize_error") = true,
           R"pydoc(
           Helper function for :meth:`truncate_singular_values`.

           Parameters
           ----------
           S_np : 1D numpy array of float
               A numpy array of singular values S[i]
           qdims : 1D numpy array of float
               A numpy array of the quantum dimensions. ``None`` means all qdims are one.
           chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error
               Constraints for truncation. See :func:`cyten.tensors.truncate_singular_values`.

           Returns
           -------
           mask : 1D numpy array of bool
               A boolean mask, indicating that ``S_np[mask]`` should be kept
           err : float
               The truncation error ``norm(S_discard) == norm(S - S_keep)``.
           new_norm
               The norm ``norm(S_keep)`` of the approximation.
           )pydoc")
      .def("zero_data",
           &TensorBackend::zero_data,
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("dtype"),
           py::arg("device"),
           py::arg("all_blocks") = false,
           R"pydoc(
           Data for a zero tensor.

           Parameters
           ----------
           all_blocks: bool
               Some specific backends can omit zero blocks ("sparsity").
               By default (``False``), omit them if possible.
               If ``True``, force all blocks to be created, with zero entries.
           )pydoc")
      .def("zero_diagonal_data",
           &TensorBackend::zero_diagonal_data,
           py::arg("co_domain"),
           py::arg("dtype"),
           py::arg("device"))
      .def(
        "zero_mask_data", &TensorBackend::zero_mask_data, py::arg("large_leg"), py::arg("device"))
      .def("is_real",
           &TensorBackend::is_real,
           py::arg("a"),
           R"pydoc(
           If the Tensor is comprised of real numbers.

           Complex numbers with small or zero imaginary part still cause a `False` return.
           )pydoc")
      .def("save_hdf5",
           &TensorBackend::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &TensorBackend::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));

    // Nested Data type under TensorBackend for Python parity with BlockBackend.BlockCls style.
    tensor_backend.attr("Data") = data_cls;

    m.def("conventional_leg_order",
          py::overload_cast<py::object, py::object>(&conventional_leg_order),
          py::arg("tensor_or_codomain"),
          py::arg("domain") = py::none(),
          R"pydoc(
          The conventional order of legs.
          )pydoc");

    m.def(
      "get_same_backend",
      [](py::args objs, py::kwargs kwargs) {
          std::string error_msg = "Incompatible backends.";
          if (kwargs.contains("error_msg"))
              error_msg = kwargs["error_msg"].cast<std::string>();
          std::vector<py::object> vec;
          vec.reserve(objs.size());
          for (auto const& o : objs)
              vec.emplace_back(py::reinterpret_borrow<py::object>(o));
          return get_same_backend(vec, std::move(error_msg));
      },
      R"pydoc(
      If the given object have the same backend, return it. Raise otherwise.
      )pydoc");
}

} // namespace cyten
