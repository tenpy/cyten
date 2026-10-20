#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/no_symmetry.h"
#include "py_lasyten_pybind11.h"

#include "symmetries/casters.hpp"

#include "tools/hdf5_bind.h"
#include <lasyten/symmetries/factors/no_symmetry.h>

namespace lasyten {

void
bind_no_symmetry(py::module_& m)
{
    py::class_<NoSymmetry, AbelianGroup, py::smart_holder>(
      m, "NoSymmetry", DOC(lasyten, NoSymmetry))
      .def(py::init<>())
      .def_static("from_hdf5",
                  lasyten::hdf5::wrap_from_hdf5<NoSymmetry>(),
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace lasyten
