#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/zn.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/zn.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_zn(py::module_& m)
{
    py::class_<ZN, AbelianGroup, py::smart_holder>(m, "ZN", DOC(lasyten, ZN))
      .def(py::init<int, std::optional<std::string>, bool>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_readonly("N", &ZN::N)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<ZN>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
