#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/mask.h>

#include "../py_cyten_pybind11.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <optional>
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

    cls.def(py::init<TensorBackend::DataPtr,
                     py::object,
                     py::object,
                     std::optional<bool>,
                     TensorBackend::Ptr,
                     py::object>(),
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

    cls.def("test_sanity", &Mask::test_sanity, "Perform sanity checks.");

    cls.def_static(
      "from_eye",
      &Mask::from_eye,
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
      &Mask::from_block_mask,
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
      &Mask::from_DiagonalTensor,
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
      &Mask::from_indices,
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
      &Mask::from_random,
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
      &Mask::from_zero,
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
                   "Import Mask from hdf5");

    cls.def("save_hdf5",
            &Mask::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            "Export Mask to hdf5 such that it can be re-imported with from_hdf5");

    cls.def("as_dtype", &Mask::as_dtype, py::arg("dtype"));

    cls.def(
      "as_SymmetricTensor",
      [](Mask& self, bool guarantee_copy, std::optional<std::string> warning, std::optional<Dtype> dtype) {
          if (dtype.has_value()) {
              return self.as_SymmetricTensor(guarantee_copy, warning, *dtype);
          }
          return self.as_SymmetricTensor(guarantee_copy, warning);
      },
      py::arg("guarantee_copy") = false,
      py::arg("warning") = py::none(),
      py::arg("dtype") = Dtype::Complex128);

    cls.def("as_DiagonalTensor",
            &Mask::as_DiagonalTensor,
            py::arg("dtype") = Dtype::Complex128);

    cls.def("as_block_mask", &Mask::as_block_mask);
    cls.def("as_numpy_mask", &Mask::as_numpy_mask);

    cls.def("all", &Mask::all, "If the mask keeps all basis elements");
    cls.def("any", &Mask::any, "If the mask keeps any basis elements");

    cls.def("copy",
            &Mask::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none());

    // Override Tensor.dagger / hc properties (which delegate to the Python free function).
    cls.def_property_readonly("dagger", &Mask::dagger);
    cls.def_property_readonly("hc", &Mask::dagger);

    cls.def("_get_item", &Mask::_get_item, py::arg("idx"));

    cls.def("logical_not", &Mask::logical_not, "Alias for :meth:`orthogonal_complement`");
    cls.def("orthogonal_complement",
            &Mask::orthogonal_complement,
            "The \"opposite\" Mask, that keeps exactly what self discards and vv.");

    cls.def("move_to_device", &Mask::move_to_device, py::arg("device"));

    cls.def("to_backend",
            &Mask::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none());

    cls.def(
      "to_dense_block",
      [](Mask& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false);

    cls.def(
      "to_numpy",
      [](Mask& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
          return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("numpy_dtype") = py::none(),
      py::arg("understood_braiding") = false);

    cls.def("_binary_operand",
            &Mask::_binary_operand,
            py::arg("other"),
            py::arg("func"),
            py::arg("operand"),
            py::arg("return_NotImplemented") = true);

    cls.def("_unary_operand", &Mask::_unary_operand, py::arg("func"));

    cls.def("__bool__", [](Mask&) {
        throw py::type_error("The truth value of a Mask is ambiguous. Use a.any() or a.all()");
    });

    cls.def("__invert__",
            [](Mask& self) { return self._unary_operand(py::module_::import("operator").attr("invert")); });

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
