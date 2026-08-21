#include "py_cyten_pybind11.h"
#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/su2.h"

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
                                                 DOC(cyten, SU2));

    cls.def(py::init<std::optional<std::string>>(), py::arg("descriptive_name") = py::none())
      .def_static(
        "from_hdf5", &SU2::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));

    // Class-level convenience sectors (match Python ``SU2.spin_half`` etc.).
    cls.attr("spin_zero") = SU2::spin_zero;
    cls.attr("spin_half") = SU2::spin_half;
    cls.attr("spin_one") = SU2::spin_one;
}

} // namespace cyten
