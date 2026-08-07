#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/ising_anyon_category.h>

namespace cyten {

void
bind_ising_anyon_category(py::module_& m)
{
    py::class_<IsingAnyonCategory, SymmetryFactor, py::smart_holder> cls(m,
                                                                         "IsingAnyonCategory",
                                                                         R"pydoc(
                                                                         Category describing Ising anyons.
                                                                         )pydoc");
    cls.def(py::init<int>(), py::arg("nu") = 1)
      .def_static("from_hdf5",
                  &IsingAnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("vacuum") = IsingAnyonCategory::vacuum;
    cls.attr("sigma") = IsingAnyonCategory::sigma;
    cls.attr("psi") = IsingAnyonCategory::psi;
    cls.def_readonly("nu", &IsingAnyonCategory::nu);
}

} // namespace cyten
