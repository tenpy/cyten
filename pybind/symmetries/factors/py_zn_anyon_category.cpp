#include "py_cyten_pybind11.h"
#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn_anyon_category.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/zn_anyon_category.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_zn_anyon_category(py::module_& m)
{
    py::class_<ZNAnyonCategory, SymmetryFactor, py::smart_holder>(m,
                                                                  "ZNAnyonCategory",
                                                                  DOC(cyten, ZNAnyonCategory))
      .def(py::init<int, int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("n"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &ZNAnyonCategory::N)
      .def_readonly("n", &ZNAnyonCategory::n)
      .def_static("from_hdf5",
                  &ZNAnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
