#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/quantum_double_zn_anyon_category.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_quantum_double_zn_anyon_category(py::module_& m)
{
    py::class_<QuantumDoubleZNAnyonCategory, SymmetryFactor, py::smart_holder>(
      m,
      "QuantumDoubleZNAnyonCategory",
      R"pydoc(
      Doubled abelian anyon category :math:`D(Z_N)`.
      )pydoc")
      .def(py::init<int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &QuantumDoubleZNAnyonCategory::N);
}

} // namespace cyten
