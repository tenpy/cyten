#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/mask.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"

#include "../py_cyten_pybind11.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

namespace {

std::optional<std::vector<std::variant<int64, std::string>>>
optional_leg_order(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<std::variant<int64, std::string>> out;
    for (auto item : to_iterable(obj)) {
        if (py::isinstance<py::str>(item)) {
            out.emplace_back(item.cast<std::string>());
        } else {
            out.emplace_back(item.cast<int64>());
        }
    }
    return out;
}

} // namespace

void
bind_tensors_mask(py::module_& m)
{
    py::class_<Mask, Tensor, py::smart_holder> cls(m, "Mask");
    cls.doc() = R"pydoc(
A boolean mask that can be used to project or enlarge a leg.

Masks come in two versions: projections and inclusions. A projection Mask has a single leg, the
:attr:`large_leg` in its domain and maps it to a single leg, the :attr:`small_leg` in the
codomain. An inclusion Mask is the dagger of this projection Mask and maps from the small leg
in the domain to the large leg in the codomain::

    |         ║                 │
    |      ┏━━┷━━┓           ┏━━┷━━┓
    |      ┃ M_p ┃    OR     ┃ M_i ┃
    |      ┗━━┯━━┛           ┗━━┯━━┛
    |         │                 ║

A Mask places restrictions on the basis order of the respective legs. For a projection Mask,
the kept basis elements from the large leg need to appear in their original order in the small
leg. Analogously, for an inclusion, the basis elements from the small leg need to be embedded
into the large leg in their original order. This restricts
the :attr:`~cyten.linalg.ElementarySpace.basis_perm` of the legs, see notes below.
Most classmethods that are used to build Masks take care of this for you.

Attributes
----------
is_projection: bool
    If the Mask is a projection or inclusion map (see class docstring above).

Parameters
----------
data
    The numerical data (i.e. boolean flags) comprising the mask. type is backend-specific.
    Should have boolean dtype.
space_in: Space
    The single space of the domain.
    This is the large leg for projections or the small leg for inclusions.
space_out: Space
    The single space of the codomain
    This is the small leg for projections or the large leg for inclusions.
is_projection: bool, optional
    If this Mask is a projection (from large to small) map.
    Otherwise it is in inclusion map (from small to large).
    Required if ``space_in == space_out``, since it is ambiguous in that case.
backend: TensorBackend, optional
    The backend of the tensor.
labels: list[list[str | None]] | list[str | None] | None
    Specify the labels for the legs.
    Can either give two lists, one for the codomain, one for the domain.
    Or a single flat list for all legs in the order of the :attr:`legs`,
    such that ``[codomain_labels, domain_labels]`` is equivalent
    to ``[*codomain_labels, *reversed(domain_labels)]``.

Notes
-----
The :attr:`~cyten.linalg.ElementarySpace.basis_perm` of the legs is constrained by the
requirements of the Mask, and in particular *depending on the data* as follows;
The following explanation is intuitive only for a projection Mask but also applies to inclusions.
Taking the ordered set of basis elements, permuting it by the large legs basis perm, then
discarding some of them according to the mask data, and finally permuting the remaining
elements back by the (inverse) small leg perm should result in a basis of the small leg,
where the relative ordering of elements is preserved.

In code, this means ::

    ranks = self.large_leg.basis_perm[mask_in_internal_basis][self.small_leg.inverse_basis_perm]

In particular, the basis permutation of the small leg is uniquely determined by the
permutation of the large leg and the mask data.

Consider the following valid example, assuming for simplicity only one one-dim. sector ::

    large_leg_perm = [2, 4, 0, 1, 3]
    mask_in_internal_basis = [True, True, False, True, False]
    # mask_in_public_basis = [False, True, True, False, True]
    small_leg_perm = [1, 2, 0]
    small_leg_perm_inv = [2, 0, 1]

Which maps an ordered basis as follows ::
    {e0, e1, e2, e3, e4}
    ---large_leg_perm--> {e2, e4, e0, e1, e3}
    ---mask_in_internal_basis--> {e2, e4, e1}
    ---small_leg_perm_inv--> {e1, e2, e4}

Such that the result is ordered.
)pydoc";

    cls.def(py::init([](TensorBackend::DataPtr data,
                        py::object space_in_obj,
                        py::object space_out_obj,
                        std::optional<bool> is_projection,
                        TensorBackend::Ptr backend,
                        py::object labels) {
                auto space_in = py_as_space_leg(space_in_obj);
                auto space_out = py_as_space_leg(space_out_obj);
                bool proj = false;
                if (!is_projection.has_value()) {
                    if (space_in->Space::dim == space_out->Space::dim) {
                        throw std::invalid_argument(
                          "Need to specify is_projection for equal spaces.");
                    }
                    proj = space_in->Space::dim > space_out->Space::dim;
                } else {
                    proj = *is_projection;
                }
                auto init = parse_tensor_init(py::make_tuple(space_out_obj),
                                              py::make_tuple(space_in_obj),
                                              std::move(backend),
                                              labels);
                auto device_s = init.backend->get_device_from_data(data);
                return std::make_shared<Mask>(std::move(data),
                                              space_in,
                                              space_out,
                                              proj,
                                              init.backend,
                                              init.symmetry,
                                              init.labels,
                                              std::move(device_s));
            }),
            py::arg("data"),
            py::arg("space_in"),
            py::arg("space_out"),
            py::arg("is_projection") = py::none(),
            py::arg("backend") = nullptr,
            py::arg("labels") = py::none());

    cls.def_readonly_static("_forbidden_dtypes", &Mask::_forbidden_dtypes);

    cls.def_readwrite("is_projection", &Mask::is_projection);

    cls.def_property(
      "data",
      [](Mask& self) -> py::object {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](Mask& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data = NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
          } else {
              self.data = obj.cast<TensorBackend::DataPtr>();
          }
      });

    cls.def_property_readonly("large_leg", &Mask::large_leg);
    cls.def_property_readonly("small_leg", &Mask::small_leg);

    cls.def("test_sanity",
            &Mask::test_sanity,
            R"pydoc(
            Perform sanity checks.
            )pydoc");

    cls.def_static(
      "from_eye",
      [](py::object leg,
         bool is_projection,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return Mask::from_eye(
            py_as_space_leg(leg), is_projection, init.backend, init.labels, device);
      },
      py::arg("leg"),
      py::arg("is_projection") = true,
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
The identity map as a Mask, i.e. the mask that keeps all states and discards none.

Parameters
----------
leg : ElementarySpace
    The single leg for the Mask, equal to both its small and large leg.
is_projection, backend, labels
    Arguments, like for constructor of :class:`Mask`.

See Also
--------
from_zero
    The projection Mask that discards all states and keeps none.
)pydoc");

    cls.def_static(
      "from_block_mask",
      [](py::object block_mask,
         py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          auto block = init.backend->block_backend->as_block(block_mask, Dtype::Bool, device);
          return Mask::from_block_mask(
            block, py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      py::arg("block_mask"),
      py::arg("large_leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
Create a projection Mask from a boolean block.

To get the related inclusion Mask, use :func:`dagger`.

The small leg of the projection is fully determined by the large leg and by the boolean
data. In particular, its basis permutation is such that the kept basis elements from the large
leg appear in order.

Parameters
----------
block_mask: Block
    A boolean Block indicating for each basis element of the public basis, if it is kept.
large_leg: Space
    The large leg, in the domain of the projection
backend, labels
    Arguments, like for the constructor
)pydoc");

    cls.def_static(
      "from_DiagonalTensor",
      [](py::object diag) { return Mask::from_DiagonalTensor(diag.cast<DiagonalTensorCPtr>()); },
      py::arg("diag"),
      R"pydoc(
Create a projection Mask from a boolean DiagonalTensor.

The resulting mask keeps exactly those basis elements for which the entry of `diag` is
``True``. To get the related inclusion Mask, use the :func:`dagger`.

The small leg of the projection is fully determined by the large leg and by `diag`.
In particular, its basis permutation is such that those basis elements from the large leg
that are kept appear in order.
)pydoc");

    cls.def_static(
      "from_indices",
      [](py::object indices,
         py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          return Mask::from_indices(
            indices, py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      py::arg("indices"),
      py::arg("large_leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
Create a projection Mask from the indices that are kept.

To get the related inclusion Mask, use :func:`dagger`.

The small leg of the projection is fully determined by the large leg and by the `indices`.
In particular, its basis permutation is such that those basis elements from the large leg
that are kept appear in order.

Parameters
----------
indices
    Valid index/indices for a 1D numpy array. The elements of the public basis of
    `large_leg` with these indices are kept by the projection.
large_leg, backend, labels
    Same as for :meth:`Mask.__init__`.
)pydoc");

    cls.def_static(
      "from_random",
      [](py::object large_leg,
         py::object small_leg,
         TensorBackend::Ptr backend,
         float64 p_keep,
         int64 min_keep,
         py::object labels,
         std::optional<std::string> device,
         py::object np_random) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          Space::Ptr small;
          if (!small_leg.is_none()) {
              small = py_as_space_leg(small_leg);
          }
          return Mask::from_random(py_as_space_leg(large_leg),
                                   small,
                                   init.backend,
                                   p_keep,
                                   min_keep,
                                   init.labels,
                                   device,
                                   np_random);
      },
      py::arg("large_leg"),
      py::arg("small_leg") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("p_keep") = 0.5,
      py::arg("min_keep") = 0,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      py::arg("np_random") = py::none(),
      R"pydoc(
Create a random projection Mask.

To get the related inclusion Mask, use :func:`dagger`.

Parameters
----------
large_leg: Space
    The large leg, in the domain of the projection
small_leg: Space, optional
    The small leg. If given, must be a subspace of the `large_leg` with compatible basis
    order (see notes in class docstring of :class:`Mask`).
    If ``None``, a small leg is randomly generated, according to `p_keep` and `min_keep`.
backend, labels
    Arguments, like for the constructor
p_keep: float, optional
    If `small_leg` is not given, the probability that any single sector is kept.
    Is ignored if `small_leg` is given, since it determines the number of kept sectors.
min_keep: int, optional
    If `small_leg` is not given, the minimum number of sectors kept.
    Is ignored of `small_leg` is given.
)pydoc");

    cls.def_static(
      "from_zero",
      [](py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          return Mask::from_zero(py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      py::arg("large_leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
The zero projection Mask, that discards all states and keeps none.

To get the related inclusion Mask, use :func:`dagger`.

Parameters
----------
large_leg: Space
    The large leg, in the domain of the projection
backend, labels
    Arguments, like for the constructor
device: str
    The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
    the block backend.

See Also
--------
from_eye
    The projection (or inclusion) Mask that keeps all states
)pydoc");

    cls.def_static("from_hdf5",
                   &Mask::from_hdf5,
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   R"pydoc(
                   Import Mask from hdf5
                   )pydoc");

    cls.def("save_hdf5",
            &Mask::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            R"pydoc(
            Export Mask to hdf5 such that it can be re-imported with from_hdf5
            )pydoc");

    cls.def("as_dtype",
            &Mask::as_dtype,
            py::arg("dtype"),
            R"pydoc(
            Convert to a tensor of the given dtype on the same device.

            Parameters
            ----------
            dtype: Dtype
                The dtype of the result.
            )pydoc");

    cls.def(
      "as_SymmetricTensor",
      [](Mask& self,
         bool guarantee_copy,
         std::optional<std::string> warning,
         std::optional<Dtype> dtype) {
          if (dtype.has_value()) {
              return self.as_SymmetricTensor(guarantee_copy, warning, *dtype);
          }
          return self.as_SymmetricTensor(guarantee_copy, warning);
      },
      py::arg("guarantee_copy") = false,
      py::arg("warning") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      R"pydoc(
      Convert to a :class:`SymmetricTensor`, if possible.

      Parameters
      ----------
      guarantee_copy : bool
          If already a SymmetricTensor, we do *not* make a copy by default.
          Set this flag to ``True`` to guarantee a copy.
      warning : str, optional
          If given, and if the conversion is non-trivial (i.e. if it was not already a
          SymmetricTensor to begin with), a warning with this text is issued.
      )pydoc");

    cls.def("as_DiagonalTensor", &Mask::as_DiagonalTensor, py::arg("dtype") = Dtype::Complex128);

    cls.def("as_block_mask", &Mask::as_block_mask);
    cls.def("as_numpy_mask", &Mask::as_numpy_mask);

    cls.def("all",
            &Mask::all,
            R"pydoc(
            If the mask keeps all basis elements
            )pydoc");
    cls.def("any",
            &Mask::any,
            R"pydoc(
            If the mask keeps any basis elements
            )pydoc");

    cls.def("copy",
            &Mask::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none(),
            R"pydoc(
            Copy the tensor.

            Parameters
            ----------
            deep: bool
                If the copy should be deep. A shallow copy is a new instance with the same data.
            device: str, optional
                The device for the result. Per default, use the same device as `self`.
            dtype: Dtype, optional
                The dtype of the result. Per default, use the same dtype as `self`.
            )pydoc");

    // Override Tensor.dagger / hc properties (which delegate to the Python free function).
    cls.def_property_readonly("dagger",
                              &Mask::dagger,
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
    cls.def_property_readonly("hc",
                              &Mask::dagger,
                              R"pydoc(
                              The :func:`dagger`
                              )pydoc");

    cls.def("_get_item",
            &Mask::_get_item,
            py::arg("idx"),
            R"pydoc(
            Implementation of :meth:`__getitem__`.

            Can assume we have one non-negative integer index per leg.
            )pydoc");

    cls.def("logical_not",
            &Mask::logical_not,
            R"pydoc(
            Alias for :meth:`orthogonal_complement`
            )pydoc");
    cls.def("orthogonal_complement",
            &Mask::orthogonal_complement,
            R"pydoc(
            The "opposite" Mask, that keeps exactly what self discards and vv.
            )pydoc");

    cls.def("move_to_device",
            &Mask::move_to_device,
            py::arg("device"),
            R"pydoc(
            Move tensor to a given device, *in place*.
            )pydoc");

    cls.def("to_backend",
            &Mask::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none(),
            R"pydoc(
            Convert to a tensor with a different backend.

            Parameters
            ----------
            backend: TensorBackend
                The backend of the result.
            dtype: Dtype, optional
                The dtype of the result. Per default, use the same dtype as `self`.
            device: str, optional
                The device for the result. Per default, use the same device as `self`.
            )pydoc");

    cls.def(
      "to_dense_block",
      [](Mask& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      R"pydoc(
      Convert to a dense block of the backend, if possible.

      This corresponds to "forgetting" the symmetry structure and is only possible if the
      symmetry :attr:`Symmetry.can_be_dropped`.
      The result is a backend-specific block, e.g. a numpy array if the block backend is a
      :class:`NumpyBlockBackend` or a torch Tensor if the backend is a :class:`TorchBlockBackend`.

      Parameters
      ----------
      leg_order: list of (int | str), optional
          If given, the leg of the resulting block are permuted to match this leg order.
      dtype: Dtype, optional
          If given, the result is converted to this dtype. Per default it has the :attr:`dtype`
          of the tensor.
      understood_braiding : bool
          For symmetries with non-trivial (but symmetric) braiding, e.g. fermions, the resulting
          dense block does no longer capture the braiding statistics correctly. This means that
          :func:`permute_legs` is not consistently reproduced by e.g. ``numpy.transpose`` on
          the dense block representation. Permuting its legs would require e.g. explicit swap
          gates. When using the result, special care needs to be taken regarding the leg order.
          To avoid this pitfall, we raise an error by default. Set this flag to ``True`` to
          disable the error. It is then your responsibility to take care of leg orders and braids.
          See :mod:`cyten.testing.swap_gate_numpy` for manipulations on these dense blocks.
      )pydoc");

    cls.def(
      "to_numpy",
      [](Mask& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
          return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("numpy_dtype") = py::none(),
      py::arg("understood_braiding") = false,
      R"pydoc(
      Convert to a numpy array
      )pydoc");

    cls.def("_binary_operand",
            &Mask::_binary_operand,
            py::arg("other"),
            py::arg("func"),
            py::arg("operand"),
            py::arg("return_NotImplemented") = true,
            R"pydoc(
            Utility function for a shared implementation of binary functions.

            Parameters
            ----------
            other
                Either a bool or a Mask. If a Mask, must have same :attr:`is_projection`.
            func
                The function with signature
                ``func(self_block: Block, other_or_other_block: bool | Block) -> Block``
            operand
                A string representation of the operand, used in error messages
            return_NotImplemented
                Whether `NotImplemented` should be returned on a non-scalar and non-`Tensor` other.
            )pydoc");

    cls.def(
      "_unary_operand",
      [](Mask& self, py::function func) {
          return self._unary_operand(adapt_block_bool_unary(func, self.backend->block_backend));
      },
      py::arg("func"));

    cls.def("__bool__", [](Mask&) {
        throw py::type_error("The truth value of a Mask is ambiguous. Use a.any() or a.all()");
    });

    cls.def("__invert__", [](Mask& self) {
        return self._unary_operand(
          adapt_block_bool_unary(py::module_::import("operator").attr("invert"),
                                 self.backend->block_backend));
    });

    auto bind_bool_binop = [&](char const* name, char const* op_name, char const* operand) {
        cls.def(
          name,
          [op_name, operand](Mask& self, py::object other) {
              return self._binary_operand(
                other, py::module_::import("operator").attr(op_name), operand, true);
          },
          py::arg("other"));
    };
    bind_bool_binop("__and__", "and_", "&");
    bind_bool_binop("__rand__", "and_", "&");
    bind_bool_binop("__or__", "or_", "|");
    bind_bool_binop("__ror__", "or_", "|");
    bind_bool_binop("__xor__", "xor", "^");
    bind_bool_binop("__rxor__", "xor", "^");
    bind_bool_binop("__eq__", "eq", "==");
    bind_bool_binop("__ne__", "ne", "!=");
}

} // namespace cyten
