#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/zn_anyon_category2.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_zn_anyon_category2(py::module_& m)
{
    py::class_<ZNAnyonCategory2, SymmetryFactor, py::smart_holder>(m,
                                                                   "ZNAnyonCategory2",
                                                                   R"pydoc(
                                                                   Abelian anyon category with fusion rules corresponding to the Z_N group;
                                                                   also written as :math:`Z_N^{(n+1/2)}`. `N` must be even.
                                                                   )pydoc")
      .def(py::init<int, int, std::optional<std::string>>(),
           py::arg("N"),
           py::arg("n"),
           py::arg("descriptive_name") = py::none())
      .def_readonly("N", &ZNAnyonCategory2::N)
      .def_readonly("n", &ZNAnyonCategory2::n)
      .def_static("from_hdf5",
                  &ZNAnyonCategory2::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
