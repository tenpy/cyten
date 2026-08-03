#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/symmetry_factor.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_symmetry_factor(py::module_& m)
{
    py::class_<SymmetryFactor, BaseSymmetry, PySymmetryFactor, py::smart_holder> cls(
      m,
      "SymmetryFactor",
      R"pydoc(
Base class for symmetries that impose a block-structure on tensors

Attributes
----------
can_be_dropped: bool
    If the symmetry could be dropped to :class:`NoSymmetry` while preserving the structure.
trivial_sector: Sector
    The trivial sector of the symmetry.
group_name: str
    A readable name for the symmetry, e.g. ``'U(1)'``.
descriptive_name: str | None
    Optional additional name, e.g. ``'Sz'``.
num_sectors: int | float
    Number of sectors, or ``float('inf')``.
)pydoc");

    cls.def(py::init<FusionStyle,
                     BraidingStyle,
                     Sector,
                     std::string,
                     float64,
                     bool,
                     std::optional<std::string>,
                     bool>(),
            py::arg("fusion_style"),
            py::arg("braiding_style"),
            py::arg("trivial_sector"),
            py::arg("group_name"),
            py::arg("num_sectors"),
            py::arg("has_complex_topological_data"),
            py::arg("descriptive_name") = py::none(),
            py::arg("trivial_shift") = true);

    cls.def_readwrite("group_name", &SymmetryFactor::group_name)
      .def_readwrite("descriptive_name", &SymmetryFactor::descriptive_name)
      .def_readwrite("fusion_tensor_dtype", &SymmetryFactor::fusion_tensor_dtype);

    cls.def("is_valid_sector", &SymmetryFactor::is_valid_sector, py::arg("a"))
      .def("fusion_outcomes", &SymmetryFactor::fusion_outcomes, py::arg("a"), py::arg("b"))
      .def("__repr__", &SymmetryFactor::repr)
      .def(
        "is_equivalent_to",
        [](SymmetryFactor& self, py::object other) {
            auto Symmetry = py::module_::import("cyten.symmetries._symmetries").attr("Symmetry");
            if (py::isinstance(other, Symmetry)) {
                return other.attr("is_equivalent_to")(py::cast(self)).cast<bool>();
            }
            return self._is_equivalent_factor(other.cast<SymmetryFactor const&>());
        },
        py::arg("other"))
      .def("_is_equivalent_factor", &SymmetryFactor::_is_equivalent_factor, py::arg("other"))
      .def("as_Symmetry",
           [](py::object self) {
               // Import submodule (not package) to avoid circular import during _symmetries load.
               auto Symmetry =
                 py::module_::import("cyten.symmetries._symmetries").attr("Symmetry");
               return Symmetry(py::make_tuple(self));
           })
      .def("__str__", &SymmetryFactor::str)
      .def("__mul__",
           [](py::object self, py::object other) {
               auto mod = py::module_::import("cyten.symmetries._symmetries");
               auto Symmetry = mod.attr("Symmetry");
               py::object SymmetryFactor_py = mod.attr("SymmetryFactor");
               if (py::isinstance(other, SymmetryFactor_py)) {
                   return Symmetry(py::make_tuple(self, other));
               }
               if (py::isinstance(other, Symmetry)) {
                   py::list factors;
                   factors.append(self);
                   for (auto f : other.attr("factors")) {
                       factors.append(f);
                   }
                   return Symmetry(factors);
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("__eq__",
           [](SymmetryFactor const& self, py::object other) {
               if (!py::isinstance<SymmetryFactor>(other)) {
                   return false;
               }
               return self.equals(other.cast<SymmetryFactor const&>());
           })
      .def("save_hdf5",
           &SymmetryFactor::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"));
}

} // namespace cyten
