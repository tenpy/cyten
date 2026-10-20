#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/ising_anyon_category.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include <lasyten/symmetries/factors/ising_anyon_category.h>

namespace lasyten {

void
bind_ising_anyon_category(py::module_& m)
{
    py::class_<IsingAnyonCategory, SymmetryFactor, py::smart_holder> cls(
      m, "IsingAnyonCategory", DOC(lasyten, IsingAnyonCategory));
    cls.def(py::init<int>(), py::arg("nu") = 1)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<IsingAnyonCategory>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("vacuum") = IsingAnyonCategory::vacuum;
    cls.attr("sigma") = IsingAnyonCategory::sigma;
    cls.attr("psi") = IsingAnyonCategory::psi;
    cls.def_readonly("nu", &IsingAnyonCategory::nu);
}

} // namespace lasyten
