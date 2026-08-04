#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/su3_3_anyon_category.h>

namespace cyten {

void
bind_su3_3_anyon_category(py::module_& m)
{
    py::class_<SU3_3AnyonCategory, SymmetryFactor, py::smart_holder> cls(m,
                                                                         "SU3_3AnyonCategory",
                                                                         R"pydoc(
                                                                         :math:`SU(3)_3` anyon category.
                                                                         )pydoc");
    cls.def(py::init<>())
      .def_static("from_hdf5",
                  &SU3_3AnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("one_irrep") = SU3_3AnyonCategory::one_irrep;
    cls.attr("eight_irrep") = SU3_3AnyonCategory::eight_irrep;
    cls.attr("ten_irrep") = SU3_3AnyonCategory::ten_irrep;
    cls.attr("ten_bar_irrep") = SU3_3AnyonCategory::ten_bar_irrep;
}

} // namespace cyten
