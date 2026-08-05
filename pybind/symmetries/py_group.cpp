#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/group.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_group(py::module_& m)
{
    py::class_<Group, SymmetryFactor, PyGroup, py::smart_holder> cls(m,
                                                                     "Group",
                                                                     R"pydoc(
                                                                     Base-class for symmetries that are described by a group.

                                                                     The symmetry is given via a faithful representation on the Hilbert space.
                                                                     Notable counter-examples are fermionic parity or anyonic grading.
                                                                     )pydoc");

    cls.def(
      py::
        init<FusionStyle, Sector, std::string, float64, bool, std::optional<std::string>, bool>(),
      py::arg("fusion_style"),
      py::arg("trivial_sector"),
      py::arg("group_name"),
      py::arg("num_sectors"),
      py::arg("has_complex_topological_data"),
      py::arg("descriptive_name") = py::none(),
      py::arg("trivial_shift") = true);

    // Group defaults; subclasses may still override via trampoline.
    cls
      .def("_fusion_tensor",
           &Group::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a"),
           py::arg("Z_b"))
      .def("swap_gate", &Group::swap_gate, py::arg("a"), py::arg("b"))
      .def("qdim", &Group::qdim, py::arg("a"))
      .def("batch_qdim", &Group::batch_qdim, py::arg("a"))
      .def("topological_twist", &Group::topological_twist, py::arg("a"));
}

} // namespace cyten
