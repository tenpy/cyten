#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/su2.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_su2(py::module_& m)
{
    py::class_<SU2, Group, py::smart_holder> cls(m,
                                                 "SU2",
                                                 R"pydoc(
                                                 SU(2) symmetry.

                                                 Allowed sectors are 1D arrays ``[jj]`` of positive integers `jj` = `0`, `1`, `2`, ...
                                                 which label the spin `jj/2` irrep of SU(2).
                                                 This is for convenience so that we can work with `int` objects.
                                                 E.g. a spin-1/2 degree of freedom is represented by the sector `[1]`.
                                                 )pydoc");

    cls.def(py::init<std::optional<std::string>>(), py::arg("descriptive_name") = py::none())
      .def_static(
        "from_hdf5", &SU2::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));

    // Class-level convenience sectors (match Python ``SU2.spin_half`` etc.).
    cls.attr("spin_zero") = SU2::spin_zero;
    cls.attr("spin_half") = SU2::spin_half;
    cls.attr("spin_one") = SU2::spin_one;
}

} // namespace cyten
