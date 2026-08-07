#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/u1.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_u1(py::module_& m)
{
    py::class_<U1, AbelianGroup, py::smart_holder>(m,
                                                   "U1",
                                                   R"pydoc(
                                                   U(1) symmetry.

                                                   Allowed sectors are 1D arrays with a single integer entry.
                                                   ..., `[-2]`, `[-1]`, `[0]`, `[1]`, `[2]`, ...
                                                   )pydoc")
      .def(py::init<std::optional<std::string>, bool>(),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_static(
        "from_hdf5", &U1::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));
}

} // namespace cyten
