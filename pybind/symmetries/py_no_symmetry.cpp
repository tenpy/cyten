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
      .def(py::init<>());
}

} // namespace cyten
