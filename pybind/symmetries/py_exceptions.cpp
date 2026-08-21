#include "../doc_plus.h"
#include "docstrings/symmetries/exceptions.h"
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
    symmetry_error.doc() = DOC(cyten, SymmetryError);

    auto& braid_err = py::register_exception<BraidChiralityUnspecifiedError>(
      m, "BraidChiralityUnspecifiedError", symmetry_error);
    braid_err.doc() = DOC(cyten, BraidChiralityUnspecifiedError);
}

} // namespace cyten
