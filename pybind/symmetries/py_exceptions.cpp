#include "py_cyten_pybind11.h"

#include <cyten/symmetries/exceptions.h>

namespace cyten {

void
bind_symmetries_exceptions(py::module_& m)
{
    // Docstrings attached after registration (register_exception has no docstring arg in pybind11
    // 3).
    auto& symmetry_error =
      py::register_exception<SymmetryError>(m, "SymmetryError", PyExc_Exception);
    symmetry_error.doc() =
      R"pydoc(
      An exception that is raised whenever something is not possible or not allowed due to symmetry
      )pydoc";

    auto& braid_err = py::register_exception<BraidChiralityUnspecifiedError>(
      m, "BraidChiralityUnspecifiedError", symmetry_error);
    braid_err.doc() =
      R"pydoc(
      An exception that is raised whenever a braid chirality should be specified but wasn't.
      )pydoc";
}

} // namespace cyten
