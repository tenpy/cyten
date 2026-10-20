#include "../doc_plus.h"
#include "docstrings/symmetries/exceptions.h"
#include "py_lasyten_pybind11.h"

#include <lasyten/symmetries/exceptions.h>

namespace lasyten {

void
bind_symmetries_exceptions(py::module_& m)
{
    // Docstrings attached after registration (register_exception has no docstring arg in pybind11
    // 3).
    auto& symmetry_error =
      py::register_exception<SymmetryError>(m, "SymmetryError", PyExc_Exception);
    symmetry_error.doc() = DOC(lasyten, SymmetryError);

    auto& braid_err = py::register_exception<BraidChiralityUnspecifiedError>(
      m, "BraidChiralityUnspecifiedError", symmetry_error);
    braid_err.doc() = DOC(lasyten, BraidChiralityUnspecifiedError);
}

} // namespace lasyten
