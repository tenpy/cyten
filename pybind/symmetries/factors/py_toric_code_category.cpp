#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/toric_code_category.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/toric_code_category.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_toric_code_category(py::module_& m)
{
    py::class_<ToricCodeCategory, QuantumDoubleZNAnyonCategory, py::smart_holder> cls(
      m, "ToricCodeCategory", DOC(cyten, ToricCodeCategory));
    cls.def(py::init<std::optional<std::string>>(), py::arg("descriptive_name") = py::none())
      .def_static("from_hdf5",
                  &ToricCodeCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("vacuum") = ToricCodeCategory::vacuum;
    cls.attr("electric_charge") = ToricCodeCategory::electric_charge;
    cls.attr("magnetic_flux") = ToricCodeCategory::magnetic_flux;
    cls.attr("fermion") = ToricCodeCategory::fermion;
}

} // namespace cyten
