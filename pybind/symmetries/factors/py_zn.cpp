#include "py_cyten_pybind11.h"
#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn.h"

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
                                                   DOC(cyten, ZN))
      .def(py::init<int, std::optional<std::string>, bool>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_readonly("N", &ZN::N)
      .def_static(
        "from_hdf5", &ZN::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));
}

} // namespace cyten
