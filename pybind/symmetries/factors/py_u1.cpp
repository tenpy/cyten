#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/u1.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <lasyten/symmetries/factors/u1.h>

#include "tools/hdf5_bind.h"
#include <optional>
#include <string>

namespace lasyten {

void
bind_u1(py::module_& m)
{
    py::class_<U1, AbelianGroup, py::smart_holder>(m, "U1", DOC(lasyten, U1))
      .def(py::init<std::optional<std::string>, bool>(),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<U1>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
