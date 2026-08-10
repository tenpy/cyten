#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/su2_k_anyon_category.h>

#include <string>

namespace cyten {

void
bind_su2_k_anyon_category(py::module_& m)
{
    py::class_<SU2_kAnyonCategory, SymmetryFactor, py::smart_holder> cls(m,
                                                                         "SU2_kAnyonCategory",
                                                                         R"pydoc(
                                                                         :math:`SU(2)_k` anyon category.
                                                                         )pydoc");
    cls.def(py::init<int, std::string>(), py::arg("k"), py::arg("handedness") = "left")
      .def_static("from_hdf5",
                  &SU2_kAnyonCategory::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
    cls.attr("spin_zero") = SU2_kAnyonCategory::spin_zero;
    cls.attr("spin_half") = SU2_kAnyonCategory::spin_half;
    cls.def_readonly("k", &SU2_kAnyonCategory::k)
      .def_readonly("handedness", &SU2_kAnyonCategory::handedness)
      .def_property_readonly("spin_one", [](SU2_kAnyonCategory const& self) -> py::object {
          if (!self.spin_one.has_value()) {
              return py::none();
          }
          return py::cast(*self.spin_one);
      });
}

} // namespace cyten
