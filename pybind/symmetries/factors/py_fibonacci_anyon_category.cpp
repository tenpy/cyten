#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/fibonacci_anyon_category.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/fibonacci_anyon_category.h>

#include <string>

namespace cyten {

void
bind_fibonacci_anyon_category(py::module_& m)
{
    py::class_<FibonacciAnyonCategory, SymmetryFactor, py::smart_holder> cls(
      m, "FibonacciAnyonCategory", DOC(cyten, FibonacciAnyonCategory));
    cls.def(py::init<std::string>(), py::arg("handedness") = "left")
      .def_static("from_hdf5",
                  &FibonacciAnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("vacuum") = FibonacciAnyonCategory::vacuum;
    cls.attr("tau") = FibonacciAnyonCategory::tau;
    cls.def_readonly("handedness", &FibonacciAnyonCategory::handedness);
}

} // namespace cyten
