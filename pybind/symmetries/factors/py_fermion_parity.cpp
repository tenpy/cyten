#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/fermion_parity.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_fermion_parity(py::module_& m)
{
    py::class_<FermionParity, SymmetryFactor, py::smart_holder> cls(m,
                                                                    "FermionParity",
                                                                    R"pydoc(
                                                                    Fermionic Parity.

                                                                    Allowed sectors are arrays with a single entry; either ``[0]`` (even) or ``1`` (odd).
                                                                    )pydoc");
    cls
      .def(py::init<std::optional<std::string>, bool>(),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true)
      .def_static("from_hdf5",
                  &FermionParity::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("even") = FermionParity::even;
    cls.attr("odd") = FermionParity::odd;
}

} // namespace cyten
