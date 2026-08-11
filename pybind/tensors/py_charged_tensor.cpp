#include <cyten/tensors/charged_tensor.h>

#include "../py_cyten_pybind11.h"

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
bind_tensors_charged_tensor(py::module_& m)
{
    py::class_<ChargedTensor, Tensor, py::smart_holder> cls(m, "ChargedTensor");
    cls.doc() = R"pydoc(
Tensors which are not symmetric, but carry a well defined charge.

This captures two related but slightly different concepts.
In both cases, the main component of a symmetric tensor is an invariant part, which
is a :class:`SymmetricTensor`, that has an additional hidden leg, which carries the charge.
See notes below.

If the symmetry is a group symmetry, a particular state (i.e. a vector) on the extra leg may be
specified. It is (generally) not symmetric, and thus this state is not a "tensor".
The composite object of invariant part and this `charged_state` then has a well-defined
transformation behavior under the action of the symmetry group; unlike a :class:`SymmetricTensor`,
which is invariant under the action, it transforms under the group representation associated
with the sectors of the additional leg.

Alternatively, if the symmetry has symmetric braiding (which includes all group symmetries),
we can leave the charged state unspecified and use the :class:`ChargedTensor` as a way to hide
an additional leg from algorithms.
We require the braiding to be symmetric, since otherwise the braiding behavior of the hidden
leg is ambiguous.

Parameters
----------
invariant_part:
    The symmetry-invariant part. the charge leg is the its ``domain.spaces[0]``.
charged_state: block | None
    Either ``None``, or a backend-specific block of shape ``(charge_leg.dim,)``, which specifies
    a state on the charge leg.
)pydoc";

    cls.def(py::init<py::object, py::object>(),
            py::arg("invariant_part"),
            py::arg("charged_state") = py::none());

    cls.def_readonly_static("_CHARGE_LEG_LABEL", &ChargedTensor::_CHARGE_LEG_LABEL);

    cls.def_readwrite("invariant_part", &ChargedTensor::invariant_part);
    cls.def_property(
      "charged_state",
      [](ChargedTensor& self) -> py::object {
          if (!self.charged_state) {
              return py::none();
          }
          return py::cast(self.charged_state);
      },
      [](ChargedTensor& self, py::object obj) {
          if (obj.is_none()) {
              self.charged_state = nullptr;
          } else {
              self.charged_state = obj.cast<BlockBackend::BlockPtr>();
          }
      });
    cls.def_readonly("charge_leg", &ChargedTensor::charge_leg);

    cls.def("test_sanity", &ChargedTensor::test_sanity, "Perform sanity checks.");

    cls.def_static("_parse_inv_domain",
                   &ChargedTensor::_parse_inv_domain,
                   py::arg("domain"),
                   py::arg("charge"));
    cls.def_static("_parse_inv_labels",
                   &ChargedTensor::_parse_inv_labels,
                   py::arg("labels"),
                   py::arg("codomain"),
                   py::arg("domain"));
    cls.def_static("supports_symmetry",
                   &ChargedTensor::supports_symmetry,
                   py::arg("symmetry"),
                   "If the :class:`ChargedTensor` concept is well defined for the `symmetry`.");

    cls.def_static("from_block_func",
                   &ChargedTensor::from_block_func,
                   py::arg("func"),
                   py::arg("charge"),
                   py::arg("codomain"),
                   py::arg("domain") = py::none(),
                   py::arg("charged_state") = py::none(),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("func_kwargs") = py::none(),
                   py::arg("shape_kw") = py::none(),
                   py::arg("dtype") = py::none(),
                   py::arg("device") = py::none(),
                   R"pydoc(
Create a charged tensor with inv_part from :meth:`SymmetricTensor.from_block_func`.
)pydoc");

    cls.def_static("from_dense_block",
                   &ChargedTensor::from_dense_block,
                   py::arg("block"),
                   py::arg("codomain"),
                   py::arg("domain") = py::none(),
                   py::arg("charge") = py::none(),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("dtype") = py::none(),
                   py::arg("device") = py::none(),
                   py::arg("tol") = 1e-6,
                   py::arg("understood_braiding") = false,
                   R"pydoc(
Convert a dense block of to a ChargedTensor, if possible.
)pydoc");

    cls.def_static("from_dense_block_single_sector",
                   &ChargedTensor::from_dense_block_single_sector,
                   py::arg("vector"),
                   py::arg("space"),
                   py::arg("sector"),
                   py::arg("backend") = nullptr,
                   py::arg("label") = py::none(),
                   py::arg("device") = py::none(),
                   R"pydoc(
Given a `vector` in single `space`, represent the components in a single given `sector`.
)pydoc");

    cls.def_static("from_invariant_part",
                   &ChargedTensor::from_invariant_part,
                   py::arg("invariant_part"),
                   py::arg("charged_state") = py::none(),
                   R"pydoc(
Like constructor, but deals with the case where invariant_part has only one leg.

In that case, we return a scalar if the charged_state is specified and raise otherwise.
)pydoc");

    cls.def_static("from_two_charge_legs",
                   &ChargedTensor::from_two_charge_legs,
                   py::arg("invariant_part"),
                   py::arg("state1") = py::none(),
                   py::arg("state2") = py::none(),
                   "Create a charged tensor from an invariant part with two charged legs.");

    cls.def_static("from_zero",
                   &ChargedTensor::from_zero,
                   py::arg("codomain"),
                   py::arg("domain") = py::none(),
                   py::arg("charge") = py::none(),
                   py::arg("charged_state") = py::none(),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("dtype") = Dtype::Complex128,
                   py::arg("device") = py::none(),
                   "A zero tensor.");

    cls.def_static("from_hdf5",
                   &ChargedTensor::from_hdf5,
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   "Import ChargedTensor from hdf5");
    cls.def("save_hdf5",
            &ChargedTensor::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            "Export ChargedTensor to hdf5 such that it can be re-imported with from_hdf5");

    cls.def("as_dtype", &ChargedTensor::as_dtype, py::arg("dtype"));
    cls.def("as_SymmetricTensor",
            &ChargedTensor::as_SymmetricTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none(),
            "Convert to symmetric tensor, if possible.");

    cls.def("copy",
            &ChargedTensor::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none());

    cls.def_property_readonly("dagger", &ChargedTensor::dagger);
    cls.def_property_readonly("hc", &ChargedTensor::dagger);

    cls.def("_get_item", &ChargedTensor::_get_item, py::arg("idx"));
    cls.def("move_to_device", &ChargedTensor::move_to_device, py::arg("device"));

    cls.def(
      "set_label",
      [](ChargedTensor& self, int64 pos, py::object label) -> ChargedTensor& {
          LegLabel lab = label.is_none() ? std::nullopt : std::optional<std::string>{ label.cast<std::string>() };
          self.set_label(pos, lab);
          return self;
      },
      py::arg("pos"),
      py::arg("label"));

    cls.def(
      "set_labels",
      [](ChargedTensor& self, py::object labels) -> ChargedTensor& {
          self.set_labels(Tensor::_init_parse_labels(labels, self.codomain, self.domain));
          return self;
      },
      py::arg("labels"));

    cls.def("to_backend",
            &ChargedTensor::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none());

    cls.def(
      "to_dense_block",
      [](ChargedTensor& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false);

    cls.def("to_dense_block_single_sector",
            &ChargedTensor::to_dense_block_single_sector,
            R"pydoc(
Return the components associated with a single sector.

Assumes a single-leg tensor living in a single sector and returns its components within
that sector.
)pydoc");
}

} // namespace cyten
