#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/fermion_number.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/fermion_number.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_fermion_number(py::module_& m)
{
    py::class_<FermionNumber, SymmetryFactor, py::smart_holder>(
      m, "FermionNumber", DOC(cyten, FermionNumber))
      .def(py::init<std::optional<std::string>, bool>(),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_static("from_hdf5",
                  &FermionNumber::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
