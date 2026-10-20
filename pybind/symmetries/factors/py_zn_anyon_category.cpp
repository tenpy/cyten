#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn_anyon_category.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/zn_anyon_category.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_zn_anyon_category(py::module_& m)
{
    py::class_<ZNAnyonCategory, SymmetryFactor, py::smart_holder>(
      m, "ZNAnyonCategory", DOC(lasyten, ZNAnyonCategory))
      .def(py::init<int, int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("n"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &ZNAnyonCategory::N)
      .def_readonly("n", &ZNAnyonCategory::n)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ZNAnyonCategory>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
