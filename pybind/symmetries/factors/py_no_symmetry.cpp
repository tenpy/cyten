#include "py_cyten_pybind11.h"
#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/no_symmetry.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/no_symmetry.h>

namespace cyten {

void
bind_no_symmetry(py::module_& m)
{
    py::class_<NoSymmetry, AbelianGroup, py::smart_holder>(m,
                                                           "NoSymmetry",
                                                           DOC(cyten, NoSymmetry))
      .def(py::init<>())
      .def_static("from_hdf5",
                  &NoSymmetry::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
