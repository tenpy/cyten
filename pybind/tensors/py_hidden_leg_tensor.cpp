#include <cyten/tensors/hidden_leg_tensor.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include "docstrings/tensors/hidden_leg_tensor.h"

#include <pybind11/stl.h>

#include <string>
#include <variant>
#include <vector>

namespace cyten {

void
bind_tensors_hidden_leg_tensor(py::module_& m)
{
    py::class_<HiddenLegTensor, SymmetricTensor, py::smart_holder> cls(m, "HiddenLegTensor");
    cls.doc() = DOC(cyten, HiddenLegTensor);

    cls.def(py::init([](py::object tensor, py::object which_legs) {
                auto tens = tensor.cast<Tensor::Ptr>();
                std::vector<std::variant<int64, std::string>> legs;
                for (auto item : which_legs) {
                    if (py::isinstance<py::str>(item)) {
                        legs.emplace_back(item.cast<std::string>());
                    } else {
                        legs.emplace_back(item.cast<int64>());
                    }
                }
                return HiddenLegTensor::from_tensor(std::move(tens), std::move(legs));
            }),
            py::arg("tensor"),
            py::arg("which_legs"),
            "Construct a HiddenLegTensor by hiding the given legs (prefixes '!' to their labels).");

    cls.def_static(
      "from_tensor",
      [](py::object tensor, py::object which_legs) {
          auto tens = tensor.cast<Tensor::Ptr>();
          std::vector<std::variant<int64, std::string>> legs;
          for (auto item : which_legs) {
              if (py::isinstance<py::str>(item)) {
                  legs.emplace_back(item.cast<std::string>());
              } else {
                  legs.emplace_back(item.cast<int64>());
              }
          }
          return HiddenLegTensor::from_tensor(std::move(tens), std::move(legs));
      },
      py::arg("tensor"),
      py::arg("which_legs"),
      DOC(cyten, HiddenLegTensor, from_tensor));

    cls.def_static("is_hidden_leg_label",
                   [](py::object label) {
                       if (label.is_none()) {
                           return false;
                       }
                       return HiddenLegTensor::is_hidden_leg_label(label.cast<std::string>());
                   },
                   py::arg("label"),
                   DOC(cyten, HiddenLegTensor, is_hidden_leg_label));

    cls.def("hidden_leg_idcs",
            &HiddenLegTensor::hidden_leg_idcs,
            DOC(cyten, HiddenLegTensor, hidden_leg_idcs));
    cls.def("public_leg_idcs",
            &HiddenLegTensor::public_leg_idcs,
            DOC(cyten, HiddenLegTensor, public_leg_idcs));
    cls.def("unhide_legs", &HiddenLegTensor::unhide_legs, DOC(cyten, HiddenLegTensor, unhide_legs));

    cls.def("test_sanity", &HiddenLegTensor::test_sanity);
    cls.def_property_readonly("dagger", &HiddenLegTensor::dagger);
    cls.def_property_readonly("hc", &HiddenLegTensor::dagger);

    cls.def(
      "as_SymmetricTensor",
      [](HiddenLegTensor& self, bool guarantee_copy, py::object warning) {
          std::optional<std::string> warn;
          if (!warning.is_none()) {
              warn = warning.cast<std::string>();
          }
          return self.as_SymmetricTensor(guarantee_copy, warn);
      },
      py::arg("guarantee_copy") = false,
      py::arg("warning") = py::none(),
      DOC(cyten, HiddenLegTensor, as_SymmetricTensor));

    cls.def_static(
      "from_hdf5",
      &HiddenLegTensor::from_hdf5,
      py::arg("hdf5_loader"),
      py::arg("h5gr"),
      py::arg("subpath"),
      DOC(cyten, HiddenLegTensor, from_hdf5));
    cls.def("save_hdf5",
            &HiddenLegTensor::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            DOC(cyten, HiddenLegTensor, save_hdf5));
}

} // namespace cyten
