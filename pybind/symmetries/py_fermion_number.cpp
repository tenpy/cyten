#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/fermion_number.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_fermion_number(py::module_& m)
{
    py::class_<FermionNumber, SymmetryFactor, py::smart_holder>(m,
                                                                "FermionNumber",
                                                                R"pydoc(
                                                                Conserves a fermionic particle number.

                                                                This is essentially U(1), but with a braid that encodes fermionic exchange statistics.
                                                                Allowed sectors are arrays with a single integer entry.
                                                                )pydoc")
      .def(py::init<std::optional<std::string>, bool>(),
           py::arg("descriptive_name") = py::none(),
           py::arg("trivial_shift") = true);
}

} // namespace cyten
