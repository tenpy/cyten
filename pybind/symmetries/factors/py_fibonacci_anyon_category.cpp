#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/fibonacci_anyon_category.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/fibonacci_anyon_category.h>

#include "tools/hdf5_bind.h"
#include <string>

namespace lasyten {

void
bind_fibonacci_anyon_category(py::module_& m)
{
    py::class_<FibonacciAnyonCategory, SymmetryFactor, py::smart_holder> cls(
      m, "FibonacciAnyonCategory", DOC(lasyten, FibonacciAnyonCategory));
    cls.def(py::init<std::string>(), py::arg("handedness") = "left")
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<FibonacciAnyonCategory>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("vacuum") = FibonacciAnyonCategory::vacuum;
    cls.attr("tau") = FibonacciAnyonCategory::tau;
    cls.def_readonly("handedness", &FibonacciAnyonCategory::handedness);
}

} // namespace lasyten
