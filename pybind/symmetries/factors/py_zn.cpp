#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/zn.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_zn(py::module_& m)
{
    py::class_<ZN, AbelianGroup, py::smart_holder>(m,
                                                   "ZN",
                                                   R"pydoc(
                                                   Z_N symmetry.

                                                   Allowed sectors are 1D arrays with a single integer entry between `0` and `N-1`.
                                                   `[0]`, `[1]`, ..., `[N-1]`
                                                   )pydoc")
      .def(py::init<int, std::optional<std::string>, bool>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_readonly("N", &ZN::N)
      .def_static(
        "from_hdf5", &ZN::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));
}

} // namespace cyten
