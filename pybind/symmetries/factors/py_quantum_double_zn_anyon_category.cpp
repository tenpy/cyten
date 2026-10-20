#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/quantum_double_zn_anyon_category.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/quantum_double_zn_anyon_category.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_quantum_double_zn_anyon_category(py::module_& m)
{
    py::class_<QuantumDoubleZNAnyonCategory, SymmetryFactor, py::smart_holder>(
      m, "QuantumDoubleZNAnyonCategory", DOC(lasyten, QuantumDoubleZNAnyonCategory))
      .def(py::init<int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &QuantumDoubleZNAnyonCategory::N)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<QuantumDoubleZNAnyonCategory>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
