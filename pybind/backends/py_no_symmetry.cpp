#include "../py_cyten_pybind11.h"

#include <cyten/backends/no_symmetry.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/block_backend/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

namespace {

/// Convert a Python BlockBackend (often a non-owning factory singleton) to shared_ptr.
std::shared_ptr<BlockBackend>
as_shared_block_backend(py::object obj)
{
    if (py::isinstance<NumpyBlockBackend>(obj)) {
        auto* p = obj.cast<NumpyBlockBackend*>();
        return NumpyBlockBackend::from_factory_shared(p->default_device);
    }
    if (py::isinstance<TorchBlockBackend>(obj)) {
        auto* p = obj.cast<TorchBlockBackend*>();
        return TorchBlockBackend::from_factory_shared(p->default_device);
    }
    // Other backends (e.g. ArrayApi): keep a non-owning shared_ptr.
    auto* raw = obj.cast<BlockBackend*>();
    return std::shared_ptr<BlockBackend>(raw, [](BlockBackend*) {});
}

/// Return Block to Python for NoSymmetry ``Data`` results.
BlockBackend::BlockPtr
py_block(TensorBackend::DataPtr d)
{
    return NoSymmetryBackend::unwrap(std::move(d));
}

TensorBackend::DataPtr
py_data(py::object obj)
{
    if (py::isinstance<TensorBackend::Data>(obj))
        return obj.cast<TensorBackend::DataPtr>();
    return NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
}

} // namespace

void
bind_no_symmetry_backend(py::module_& m)
{
    py::class_<NoSymmetryBackend::BlockData, TensorBackend::Data, py::smart_holder>(
      m, "NoSymmetryBackendBlockData")
      .def(py::init<BlockBackend::BlockPtr>(), py::arg("block"))
      .def_readwrite("block", &NoSymmetryBackend::BlockData::block);

    py::class_<NoSymmetryBackend, TensorBackend, py::smart_holder> cls(m, "NoSymmetryBackend");
    cls.doc() = R"pydoc(
Abstract base class for backends that do not enforce any symmetry.

Notes
-----
The data stored for the various tensor classes defined in ``cyten.tensors`` is::

    - ``SymmetricTensor``:
        A single Block with as many axes as there a legs on the tensor.
        Same leg order as ``Tensor.legs``, i.e. ``[*codomain, *reversed(domain)]``.

    - ``DiagonalTensor`` :
        A single 1D Block. The diagonal of the corresponding 2D block of a ``Tensor``.

    - ``Mask``:
        The bool values indicate which indices of the large leg are kept for the small leg.
)pydoc";

    cls.def(py::init([](py::object block_backend) {
                return std::make_shared<NoSymmetryBackend>(as_shared_block_backend(block_backend));
            }),
            py::arg("block_backend"));

    // Static helpers (useful for tests / debugging; not in original Python API).
    cls.def_static("wrap", &NoSymmetryBackend::wrap, py::arg("block"));
    cls.def_static("unwrap", &NoSymmetryBackend::unwrap, py::arg("data"));

    // Overrides that return Data → expose Block to Python (match current Python storage).
    cls.def(
      "act_block_diagonal_square_matrix",
      [](NoSymmetryBackend& self, py::object a, py::function block_method, py::object dtype_map) {
          return py_block(self.act_block_diagonal_square_matrix(a, block_method, dtype_map));
      },
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
      )pydoc");
    cls.def(
      "add_trivial_leg",
      [](NoSymmetryBackend& self,
         py::object a,
         int64 legs_pos,
         bool add_to_domain,
         int64 co_domain_pos,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.add_trivial_leg(
            a, legs_pos, add_to_domain, co_domain_pos, new_codomain, new_domain));
      },
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
      )pydoc");
    cls.def(
      "apply_mask_to_DiagonalTensor",
      [](NoSymmetryBackend& self, py::object tensor, py::object mask) {
          return py_block(self.apply_mask_to_DiagonalTensor(tensor, mask));
      },
      py::arg("tensor"),
      py::arg("mask"));
    cls.def(
      "combine_legs",
      [](NoSymmetryBackend& self,
         py::object tensor,
         std::vector<std::vector<int64>> leg_idcs_combine,
         std::vector<LegPipe::Ptr> pipes,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.combine_legs(
            tensor, std::move(leg_idcs_combine), std::move(pipes), new_codomain, new_domain));
      },
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
      )pydoc");
    cls.def(
      "compose",
      [](NoSymmetryBackend& self, py::object a, py::object b) {
          return py_block(self.compose(a, b));
      },
      py::arg("a"),
      py::arg("b"),
      R"pydoc(
      Assumes ``a.domain == b.codomain`` and performs contraction over those legs.

      Assumes there is at least one open leg, i.e. the codomain of `a` and the domain of `b` are
      not both empty. Assumes both input tensors are on the same device.
      )pydoc");
    cls.def(
      "copy_data",
      [](NoSymmetryBackend& self, py::object a, std::optional<std::string> device) {
          return py_block(self.copy_data(a, std::move(device)));
      },
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
      )pydoc");
    cls.def(
      "dagger",
      [](NoSymmetryBackend& self, py::object a) { return py_block(self.dagger(a)); },
      py::arg("a"),
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
      )pydoc");
    cls.def(
      "item",
      [](NoSymmetryBackend& self, py::object a) {
          // Python stores Block as tensor.data; wrap before data_item.
          return self.data_item(py_data(a.attr("data")));
      },
      py::arg("a"),
      R"pydoc(
      Convert tensor to a python scalar.

      Assumes that tensor is a scalar (i.e. has only one entry).
      )pydoc");
    cls.def(
      "data_item",
      [](NoSymmetryBackend& self, py::object a) { return self.data_item(py_data(a)); },
      py::arg("a"),
      R"pydoc(
      Assumes that data is a scalar (as defined in tensors.is_scalar).

      Return that scalar as python float or complex
      )pydoc");
    cls.def(
      "diagonal_elementwise_binary",
      [](NoSymmetryBackend& self,
         py::object a,
         py::object b,
         py::function func,
         py::dict func_kwargs,
         bool partial_zero_is_zero) {
          return py_block(
            self.diagonal_elementwise_binary(a, b, func, func_kwargs, partial_zero_is_zero));
      },
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
      )pydoc");
    cls.def(
      "diagonal_elementwise_unary",
      [](NoSymmetryBackend& self,
         py::object a,
         py::function func,
         py::dict func_kwargs,
         bool maps_zero_to_zero) {
          return py_block(
            self.diagonal_elementwise_unary(a, func, func_kwargs, maps_zero_to_zero));
      },
      py::arg("a"),
      py::arg("func"),
      py::arg("func_kwargs"),
      py::arg("maps_zero_to_zero"),
      R"pydoc(
      Return a modified copy of the data, resulting from applying an elementwise function.

      Apply ``func(block: Block, **kwargs) -> Block`` to all elements of a diagonal tensor.
      ``maps_zero_to_zero=True`` promises that ``func(zero_block) == zero_block``.
      )pydoc");
    cls.def(
      "diagonal_from_block",
      [](NoSymmetryBackend& self,
         BlockBackend::BlockPtr a,
         TensorProduct::Ptr co_domain,
         float64 tol) { return py_block(self.diagonal_from_block(std::move(a), co_domain, tol)); },
      py::arg("a"),
      py::arg("co_domain"),
      py::arg("tol"),
      R"pydoc(
      The DiagonalData from a 1D block in *internal* basis order.
      )pydoc");
    cls.def(
      "diagonal_from_sector_block_func",
      [](NoSymmetryBackend& self, py::function func, TensorProduct::Ptr co_domain) {
          return py_block(self.diagonal_from_sector_block_func(func, co_domain));
      },
      py::arg("func"),
      py::arg("co_domain"),
      R"pydoc(
      Generate diagonal data from a function.

      Signature is ``func(shape: tuple[int], coupled: Sector) -> Block``.
      Assumes all generated blocks are on the same device.
      )pydoc");
    cls.def(
      "diagonal_tensor_from_full_tensor",
      [](NoSymmetryBackend& self, py::object a, std::optional<float64> tol) {
          return py_block(self.diagonal_tensor_from_full_tensor(a, tol));
      },
      py::arg("a"),
      py::arg("tol") = 1e-12,
      R"pydoc(
      Get the DiagonalData corresponding to a tensor with two legs.

      Can assume that domain and codomain consist of the same single leg.
      )pydoc");
    cls.def(
      "diagonal_to_mask",
      [](NoSymmetryBackend& self, py::object tens) {
          auto [data, leg] = self.diagonal_to_mask(tens);
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      py::arg("tens"),
      R"pydoc(
      Convert a DiagonalTensor to a Mask.

      May assume that dtype is bool.
      Returns ``mask_data, small_leg``.
      )pydoc");
    cls.def(
      "diagonal_transpose",
      [](NoSymmetryBackend& self, py::object tens) {
          auto [leg, data] = self.diagonal_transpose(tens);
          return std::make_tuple(std::move(leg), py_block(std::move(data)));
      },
      py::arg("tens"),
      R"pydoc(
      Transpose a diagonal tensor. Also return the new leg ``tens.leg.dual``
      )pydoc");
    cls.def(
      "eigh",
      [](NoSymmetryBackend& self,
         py::object a,
         bool new_leg_dual,
         std::optional<std::string> sort) {
          auto [w, v, leg] = self.eigh(a, new_leg_dual, std::move(sort));
          return std::make_tuple(py_block(std::move(w)), py_block(std::move(v)), std::move(leg));
      },
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
      )pydoc");
    cls.def(
      "eye_data",
      [](NoSymmetryBackend& self, TensorProduct::Ptr co_domain, Dtype dtype, std::string device) {
          return py_block(self.eye_data(co_domain, dtype, std::move(device)));
      },
      py::arg("co_domain"),
      py::arg("dtype"),
      py::arg("device"),
      R"pydoc(
      Data for :meth:``SymmetricTensor.eye``.

      The result has legs ``first_legs + [l.dual for l in reversed(firs_legs)]``.
      )pydoc");
    cls.def(
      "from_dense_block",
      [](NoSymmetryBackend& self,
         BlockBackend::BlockPtr a,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         float64 tol) {
          return py_block(self.from_dense_block(std::move(a), codomain, domain, tol));
      },
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
      )pydoc");
    cls.def(
      "from_dense_block_trivial_sector",
      [](NoSymmetryBackend& self, BlockBackend::BlockPtr block, Space::Ptr leg) {
          return py_block(self.from_dense_block_trivial_sector(std::move(block), leg));
      },
      py::arg("block"),
      py::arg("leg"),
      R"pydoc(
      Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.

      Is given in the *internal* basis order.
      )pydoc");
    cls.def(
      "from_grid",
      [](NoSymmetryBackend& self,
         std::vector<std::vector<py::object>> grid,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain,
         std::vector<std::vector<int64>> left_mult_slices,
         std::vector<std::vector<int64>> right_mult_slices,
         Dtype dtype,
         std::string device) {
          return py_block(self.from_grid(std::move(grid),
                                         new_codomain,
                                         new_domain,
                                         std::move(left_mult_slices),
                                         std::move(right_mult_slices),
                                         dtype,
                                         std::move(device)));
      },
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
      )pydoc");
    cls.def(
      "from_random_normal",
      [](NoSymmetryBackend& self,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         float64 sigma,
         Dtype dtype,
         std::string device) {
          return py_block(
            self.from_random_normal(codomain, domain, sigma, dtype, std::move(device)));
      },
      py::arg("codomain"),
      py::arg("domain"),
      py::arg("sigma"),
      py::arg("dtype"),
      py::arg("device"));
    cls.def(
      "from_sector_block_func",
      [](NoSymmetryBackend& self,
         py::function func,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain) {
          return py_block(self.from_sector_block_func(func, codomain, domain));
      },
      py::arg("func"),
      py::arg("codomain"),
      py::arg("domain"),
      R"pydoc(
      Generate tensor data from a function ``func(shape: tuple[int], coupled: Sector) -> Block``.
      )pydoc");
    cls.def(
      "from_tree_pairs",
      [](NoSymmetryBackend& self,
         std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         Dtype dtype,
         std::string device) {
          return py_block(
            self.from_tree_pairs(std::move(trees), codomain, domain, dtype, std::move(device)));
      },
      py::arg("trees"),
      py::arg("codomain"),
      py::arg("domain"),
      py::arg("dtype"),
      py::arg("device"),
      R"pydoc(
      Compute the data for :meth:`SymmetricTensor.from_tree_pairs`.
      )pydoc");
    cls.def(
      "full_data_from_diagonal_tensor",
      [](NoSymmetryBackend& self, py::object a) {
          return py_block(self.full_data_from_diagonal_tensor(a));
      },
      py::arg("a"));
    cls.def(
      "full_data_from_mask",
      [](NoSymmetryBackend& self, py::object a, Dtype dtype) {
          return py_block(self.full_data_from_mask(a, dtype));
      },
      py::arg("a"),
      py::arg("dtype"),
      R"pydoc(
      May assume that the mask is a projection.
      )pydoc");
    cls.def(
      "get_device_from_data",
      [](NoSymmetryBackend& self, py::object a) { return self.get_device_from_data(py_data(a)); },
      py::arg("a"),
      R"pydoc(
      Extract the device from the data object
      )pydoc");
    cls.def(
      "get_dtype_from_data",
      [](NoSymmetryBackend& self, py::object a) { return self.get_dtype_from_data(py_data(a)); },
      py::arg("a"));
    cls.def(
      "inv_part_from_dense_block_single_sector",
      [](NoSymmetryBackend& self,
         BlockBackend::BlockPtr vector,
         Space::Ptr space,
         ElementarySpace::Ptr charge_leg) {
          return py_block(
            self.inv_part_from_dense_block_single_sector(std::move(vector), space, charge_leg));
      },
      py::arg("vector"),
      py::arg("space"),
      py::arg("charge_leg"),
      R"pydoc(
      Data for the invariant part used in ChargedTensor.from_dense_block_single_sector

      The vector is given in the *internal* basis order of `spaces`.
      )pydoc");
    cls.def(
      "linear_combination",
      [](NoSymmetryBackend& self,
         BlockBackend::Scalar a,
         py::object v,
         BlockBackend::Scalar b,
         py::object w) { return py_block(self.linear_combination(a, v, b, w)); },
      py::arg("a"),
      py::arg("v"),
      py::arg("b"),
      py::arg("w"),
      R"pydoc(
      Form the linear combinations ``a * v + b * w``.

      Assumes `v` and `w` are on the same device.
      )pydoc");
    cls.def(
      "lq",
      [](NoSymmetryBackend& self, py::object tensor, TensorProduct::Ptr new_co_domain) {
          auto [l, q] = self.lq(tensor, new_co_domain);
          return std::make_tuple(py_block(std::move(l)), py_block(std::move(q)));
      },
      py::arg("tensor"),
      py::arg("new_co_domain"),
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
      )pydoc");
    cls.def(
      "mask_binary_operand",
      [](NoSymmetryBackend& self, py::object mask1, py::object mask2, py::function func) {
          auto [data, leg] = self.mask_binary_operand(mask1, mask2, func);
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      py::arg("mask1"),
      py::arg("mask2"),
      py::arg("func"),
      R"pydoc(
      Elementwise binary function acting on two masks.

      May assume that both masks are a projection (from large to small leg)
      and that the large legs match.

      Assumes that `mask1` and `mask2` are on the same device.

      returns ``mask_data, new_small_leg``
      )pydoc");
    cls.def(
      "mask_contract_large_leg",
      [](NoSymmetryBackend& self, py::object tensor, py::object mask, int64 leg_idx) {
          auto [data, codomain, domain] = self.mask_contract_large_leg(tensor, mask, leg_idx);
          return std::make_tuple(
            py_block(std::move(data)), std::move(codomain), std::move(domain));
      },
      py::arg("tensor"),
      py::arg("mask"),
      py::arg("leg_idx"),
      R"pydoc(
      Contraction with the large leg of a Mask.

      Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
      the large leg of the mask is contracted.
      Note that the mask may be a projection to be applied to the codomain or an inclusion
      to be contracted on the domain.
      )pydoc");
    cls.def(
      "mask_contract_small_leg",
      [](NoSymmetryBackend& self, py::object tensor, py::object mask, int64 leg_idx) {
          auto [data, codomain, domain] = self.mask_contract_small_leg(tensor, mask, leg_idx);
          return std::make_tuple(
            py_block(std::move(data)), std::move(codomain), std::move(domain));
      },
      py::arg("tensor"),
      py::arg("mask"),
      py::arg("leg_idx"),
      R"pydoc(
      Contraction with the small leg of a Mask.

      Implementation of :func:`cyten.tensors._compose_with_Mask` in the case where
      the small leg of the mask is contracted.
      Note that the mask may be an inclusion to be applied to the codomain or a projection
      to be contracted on the domain.
      )pydoc");
    cls.def(
      "mask_dagger",
      [](NoSymmetryBackend& self, py::object mask) { return py_block(self.mask_dagger(mask)); },
      py::arg("mask"));
    cls.def(
      "mask_from_block",
      [](NoSymmetryBackend& self, BlockBackend::BlockPtr a, Space::Ptr large_leg) {
          auto [data, leg] = self.mask_from_block(std::move(a), large_leg);
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      py::arg("a"),
      py::arg("large_leg"),
      R"pydoc(
      Data for a *projection* Mask, and the resulting small leg, from a 1D block.

      a: 1D block, the Mask in *internal* basis order of `large_leg`.
      )pydoc");
    cls.def(
      "mask_to_diagonal",
      [](NoSymmetryBackend& self, py::object a, Dtype dtype) {
          return py_block(self.mask_to_diagonal(a, dtype));
      },
      py::arg("a"),
      py::arg("dtype"));
    cls.def(
      "mask_transpose",
      [](NoSymmetryBackend& self, py::object tens) {
          auto [s_in, s_out, data] = self.mask_transpose(tens);
          return std::make_tuple(std::move(s_in), std::move(s_out), py_block(std::move(data)));
      },
      py::arg("tens"),
      R"pydoc(
      Transpose a mask. Also return the new ``space_in`` and ``space_out``.

      Those spaces are the duals of the respective other in the old mask.
      )pydoc");
    cls.def(
      "mask_unary_operand",
      [](NoSymmetryBackend& self, py::object mask, py::function func) {
          auto [data, leg] = self.mask_unary_operand(mask, func);
          return std::make_tuple(py_block(std::move(data)), std::move(leg));
      },
      py::arg("mask"),
      py::arg("func"),
      R"pydoc(
      Elementwise function acting on a mask.

      May assume that mask is a projection (from large to small leg).
      Returns ``mask_data, new_small_leg``
      )pydoc");
    cls.def(
      "move_to_device",
      [](NoSymmetryBackend& self, py::object a, std::string device) {
          return py_block(self.move_to_device(a, std::move(device)));
      },
      py::arg("a"),
      py::arg("device"),
      R"pydoc(
      Move tensor to a given device.

      The result is *not* guaranteed to be a copy. In particular, if `a` already is on the
      target device, it is returned without modification.

      See Also
      --------
      copy_data
      )pydoc");
    cls.def(
      "mul",
      [](NoSymmetryBackend& self, BlockBackend::Scalar a, py::object b) {
          return py_block(self.mul(a, b));
      },
      py::arg("a"),
      py::arg("b"));
    cls.def(
      "outer",
      [](NoSymmetryBackend& self, py::object a, py::object b) {
          return py_block(self.outer(a, b));
      },
      py::arg("a"),
      py::arg("b"),
      R"pydoc(
      Form the outer product, or tensor product of maps.

      Assumes that `a` and `b` are on the same device.
      )pydoc");
    cls.def(
      "partial_compose",
      [](NoSymmetryBackend& self,
         py::object a,
         py::object b,
         int64 a_first_leg,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.partial_compose(a, b, a_first_leg, new_codomain, new_domain));
      },
      py::arg("a"),
      py::arg("b"),
      py::arg("a_first_leg"),
      py::arg("new_codomain"),
      py::arg("new_domain"),
      R"pydoc(
      Contract the codomain (domain) of `b` with the a part of the domain (codomain) of `a`.

      Assumes that there is at least one open leg in the domain (codomain) of the resulting
      tensor. Assumes both input tensors are on the same device.
      )pydoc");
    cls.def(
      "partial_trace",
      [](NoSymmetryBackend& self,
         py::object tensor,
         std::vector<std::pair<int64, int64>> pairs,
         std::vector<std::optional<int64>> levels) -> py::object {
          auto [data, codomain, domain] =
            self.partial_trace(tensor, std::move(pairs), std::move(levels));
          if (!codomain && !domain) {
              // Match Python: return scalar item when fully traced.
              return py::make_tuple(
                self.block_backend->item(py_block(data)), py::none(), py::none());
          }
          return py::make_tuple(py_block(std::move(data)), std::move(codomain), std::move(domain));
      },
      py::arg("tensor"),
      py::arg("pairs"),
      py::arg("levels") = py::none(),
      R"pydoc(
      Perform an arbitrary number of traces. Pairs are converted to leg idcs.

      Returns ``data, codomain, domain``.
      )pydoc");
    cls.def(
      "permute_legs",
      [](NoSymmetryBackend& self,
         py::object a,
         std::vector<int64> codomain_idcs,
         std::vector<int64> domain_idcs,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain,
         bool mixes_codomain_domain,
         std::vector<std::optional<int64>> levels,
         std::vector<std::optional<bool>> bend_right) {
          return py_block(self.permute_legs(a,
                                            std::move(codomain_idcs),
                                            std::move(domain_idcs),
                                            new_codomain,
                                            new_domain,
                                            mixes_codomain_domain,
                                            std::move(levels),
                                            std::move(bend_right)));
      },
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
      )pydoc");
    cls.def(
      "qr",
      [](NoSymmetryBackend& self, py::object a, TensorProduct::Ptr new_co_domain) {
          auto [q, r] = self.qr(a, new_co_domain);
          return std::make_tuple(py_block(std::move(q)), py_block(std::move(r)));
      },
      py::arg("a"),
      py::arg("new_co_domain"),
      R"pydoc(
      Perform a QR decomposition.

      With ``a == Q @ R``
      ``Q.domain == a.domain``, ``Q.codomain == new_codomain``
      ``R.domain == new_codomain``, ``R.codomain == a.codomain``
      )pydoc");
    cls.def(
      "scale_axis",
      [](NoSymmetryBackend& self, py::object a, py::object b, int64 leg) {
          return py_block(self.scale_axis(a, b, leg));
      },
      py::arg("a"),
      py::arg("b"),
      py::arg("leg"),
      R"pydoc(
      Scale axis ``leg`` of ``a`` with ``b``.

      Can assume ``a.get_leg_co_domain(leg) == b.leg``.
      Assumes that `a` and `b` are on the same device.
      )pydoc");
    cls.def(
      "split_legs",
      [](NoSymmetryBackend& self,
         py::object a,
         std::vector<int64> leg_idcs,
         TensorProduct::Ptr new_codomain,
         TensorProduct::Ptr new_domain) {
          return py_block(self.split_legs(a, std::move(leg_idcs), new_codomain, new_domain));
      },
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
      )pydoc");
    cls.def(
      "squeeze_legs",
      [](NoSymmetryBackend& self, py::object a, std::vector<int64> idcs) {
          return py_block(self.squeeze_legs(a, std::move(idcs)));
      },
      py::arg("a"),
      py::arg("idcs"),
      R"pydoc(
      Assume the legs at given indices are trivial and get rid of them
      )pydoc");
    cls.def(
      "svd",
      [](NoSymmetryBackend& self,
         py::object a,
         TensorProduct::Ptr new_co_domain,
         std::optional<std::string> algorithm) {
          auto [u, s, vh] = self.svd(a, new_co_domain, std::move(algorithm));
          return std::make_tuple(
            py_block(std::move(u)), py_block(std::move(s)), py_block(std::move(vh)));
      },
      py::arg("a"),
      py::arg("new_co_domain"),
      py::arg("algorithm") = py::none(),
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
      )pydoc");
    cls.def(
      "to_block_backend",
      [](NoSymmetryBackend& self,
         py::object data,
         py::object block_backend,
         std::optional<Dtype> dtype,
         std::optional<std::string> device) {
          return py_block(self.to_block_backend(
            py_data(data), as_shared_block_backend(block_backend), dtype, std::move(device)));
      },
      py::arg("data"),
      py::arg("block_backend"),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none());
    cls.def(
      "to_dtype",
      [](NoSymmetryBackend& self, py::object a, Dtype dtype) {
          return py_block(self.to_dtype(a, dtype));
      },
      py::arg("a"),
      py::arg("dtype"),
      R"pydoc(
      Cast to given dtype. No copy if already has dtype.
      )pydoc");
    cls.def(
      "trace_full",
      [](NoSymmetryBackend& self,
         py::object a,
         std::vector<int64> idcs1,
         std::vector<int64> idcs2) {
          return self.trace_full(a, std::move(idcs1), std::move(idcs2));
      },
      py::arg("a"),
      py::arg("idcs1") = std::vector<int64>{},
      py::arg("idcs2") = std::vector<int64>{});
    cls.def(
      "truncate_singular_values",
      [](NoSymmetryBackend& self,
         py::object S,
         std::optional<int64> chi_max,
         int64 chi_min,
         float64 degeneracy_tol,
         float64 trunc_cut,
         std::optional<float64> svd_min,
         bool minimize_error) {
          auto [mask, leg, err, new_norm] = self.truncate_singular_values(
            S, chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error);
          return std::make_tuple(py_block(std::move(mask)), std::move(leg), err, new_norm);
      },
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
      )pydoc");
    cls.def(
      "zero_data",
      [](NoSymmetryBackend& self,
         TensorProduct::Ptr codomain,
         TensorProduct::Ptr domain,
         Dtype dtype,
         std::string device,
         bool all_blocks) {
          return py_block(self.zero_data(codomain, domain, dtype, std::move(device), all_blocks));
      },
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
      )pydoc");
    cls.def(
      "zero_diagonal_data",
      [](NoSymmetryBackend& self, TensorProduct::Ptr co_domain, Dtype dtype, std::string device) {
          return py_block(self.zero_diagonal_data(co_domain, dtype, std::move(device)));
      },
      py::arg("co_domain"),
      py::arg("dtype"),
      py::arg("device"));
    cls.def(
      "zero_mask_data",
      [](NoSymmetryBackend& self, Space::Ptr large_leg, std::string device) {
          return py_block(self.zero_mask_data(large_leg, std::move(device)));
      },
      py::arg("large_leg"),
      py::arg("device"));
}

} // namespace cyten
