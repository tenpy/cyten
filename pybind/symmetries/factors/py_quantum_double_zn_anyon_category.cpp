#include "py_cyten_pybind11.h"
#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/quantum_double_zn_anyon_category.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/quantum_double_zn_anyon_category.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_quantum_double_zn_anyon_category(py::module_& m)
{
    py::class_<QuantumDoubleZNAnyonCategory, SymmetryFactor, py::smart_holder>(
      m,
      "QuantumDoubleZNAnyonCategory",
      DOC(cyten, QuantumDoubleZNAnyonCategory))
      .def(py::init<int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &QuantumDoubleZNAnyonCategory::N)
      .def_static("from_hdf5",
                  &QuantumDoubleZNAnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
