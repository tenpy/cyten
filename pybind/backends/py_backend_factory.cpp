#include "../py_cyten_pybind11.h"
#include "../doc_plus.h"
#include "docstrings/backends/backend_factory.h"

#include <cyten/backends/backend_factory.h>

namespace cyten {

void
bind_backend_factory(py::module_& m)
{
    m.def("get_backend",
          py::overload_cast<py::object, py::object>(&get_backend),
          py::arg("symmetry") = py::none(),
          py::arg("block_backend") = py::none(),
          DOC(cyten, get_backend));
}

} // namespace cyten
