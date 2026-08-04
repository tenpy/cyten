#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/toric_code_category.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_toric_code_category(py::module_& m)
{
    py::class_<ToricCodeCategory, QuantumDoubleZNAnyonCategory, py::smart_holder> cls(
      m,
      "ToricCodeCategory",
      R"pydoc(
      Toric code anyon category. Essentially equivalent to `QuantumDoubleZNAnyonCategory(N=2)`.
      )pydoc");
    cls.def(py::init<std::optional<std::string>>(), py::arg("descriptive_name") = py::none());
    cls.attr("vacuum") = ToricCodeCategory::vacuum;
    cls.attr("electric_charge") = ToricCodeCategory::electric_charge;
    cls.attr("magnetic_flux") = ToricCodeCategory::magnetic_flux;
    cls.attr("fermion") = ToricCodeCategory::fermion;
}

} // namespace cyten
