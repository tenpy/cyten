#include "../doc_plus.h"
#include "../py_lasyten_pybind11.h"
#include "docstrings/backends/backend_factory.h"

#include <lasyten/backends/backend_factory.h>

namespace lasyten {

void
bind_backend_factory(py::module_& m)
{
    m.def("get_backend",
          py::overload_cast<py::object, py::object>(&get_backend),
          py::arg("symmetry") = py::none(),
          py::arg("block_backend") = py::none(),
          DOC(lasyten, get_backend));
}

} // namespace lasyten
