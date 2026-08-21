#include "py_cyten_pybind11.h"
#include "../doc_plus.h"
#include "docstrings/symmetries/group.h"

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
                                                                     DOC(cyten, Group));

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
           py::arg("Z_b"),
           DOC(cyten, Group, _fusion_tensor))
      .def("swap_gate",
           &Group::swap_gate,
           py::arg("a"),
           py::arg("b"),
           doc_cpp_ref(R"pydoc(swap_gate)pydoc", "cyten::BaseSymmetry::swap_gate()"))
      .def("qdim",
           &Group::qdim,
           py::arg("a"),
           doc_cpp_ref(R"pydoc(qdim)pydoc", "cyten::BaseSymmetry::qdim()"))
      .def(
        "batch_qdim",
        [](Group const& self, SectorArray const& a) {
            return vector_f64_to_numpy(self.batch_qdim(a));
        },
        py::arg("a"),
        DOC(cyten, Group, batch_qdim))
      .def("topological_twist",
           &Group::topological_twist,
           py::arg("a"),
           DOC(cyten, Group, topological_twist));
}

} // namespace cyten
