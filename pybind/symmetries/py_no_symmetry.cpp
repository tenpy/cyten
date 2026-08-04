#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/no_symmetry.h>

namespace cyten {

void
bind_no_symmetry(py::module_& m)
{
    py::class_<NoSymmetry, AbelianGroup, py::smart_holder>(m,
                                                           "NoSymmetry",
                                                           R"pydoc(
                                                           Trivial symmetry group that doesn't do anything.

                                                           The only allowed sector is ``[0]``.
                                                           )pydoc")
      .def(py::init<>())
      .def_static("from_hdf5",
                  &NoSymmetry::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
