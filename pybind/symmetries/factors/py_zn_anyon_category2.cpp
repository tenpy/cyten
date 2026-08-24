#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn_anyon_category2.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/zn_anyon_category2.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_zn_anyon_category2(py::module_& m)
{
    py::class_<ZNAnyonCategory2, SymmetryFactor, py::smart_holder>(
      m, "ZNAnyonCategory2", DOC(cyten, ZNAnyonCategory2))
      .def(py::init<int, int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("n"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &ZNAnyonCategory2::N)
      .def_readonly("n", &ZNAnyonCategory2::n)
      .def_static("from_hdf5",
                  &ZNAnyonCategory2::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));

    m.def("semion_category", &semion_category, DOC(cyten, semion_category));
    m.def("double_semion_category", &double_semion_category, DOC(cyten, double_semion_category));
}

} // namespace cyten
