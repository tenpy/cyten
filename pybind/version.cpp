#include <cyten/version.h>

#include "cyten_pybind11.h"

namespace py = pybind11;
namespace cyten {

void
bind_version(py::module_& m)
{
    m.def("get_build_version",
          &get_build_version,
          "Get the build version",
          py::return_value_policy::reference);
}

} // namespace cyten
