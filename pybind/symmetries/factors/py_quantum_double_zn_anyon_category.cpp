#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/quantum_double_zn_anyon_category.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_quantum_double_zn_anyon_category(py::module_& m)
{
    py::class_<QuantumDoubleZNAnyonCategory, SymmetryFactor, py::smart_holder>(
      m,
      "QuantumDoubleZNAnyonCategory",
      R"pydoc(
      Doubled abelian anyon category.

      The fusion rules corresponding to the :math:`Z_N \times Z_N` group.
      The category is commonly written as :math:`D(Z_N)`.

      Allowed sectors are 1D arrays with two integers between ``0`` and ``N-1``.
      ``[0, 0]``, ``[0, 1]``, ..., ``[N-1, N-1]``.

      This is not a simple product of two :class:`ZNAnyonCategory`\ s; there are nontrivial R-symbols.
      )pydoc")
      .def(py::init<int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &QuantumDoubleZNAnyonCategory::N)
      .def_static("from_hdf5",
                  &QuantumDoubleZNAnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
