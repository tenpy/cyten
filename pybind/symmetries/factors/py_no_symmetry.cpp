#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/no_symmetry.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include <cyten/symmetries/factors/no_symmetry.h>

namespace cyten {

void
bind_no_symmetry(py::module_& m)
{
    py::class_<NoSymmetry, AbelianGroup, py::smart_holder>(m, "NoSymmetry", DOC(cyten, NoSymmetry))
      .def(py::init<>())
      .def_static("from_hdf5",
                  cyten::hdf5::wrap_from_hdf5<NoSymmetry>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
