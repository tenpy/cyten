#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/abelian_group.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_abelian_group(py::module_& m)
{
    py::class_<AbelianGroup, Group, PyAbelianGroup, py::smart_holder> cls(m,
                                                                          "AbelianGroup",
                                                                          R"pydoc(
                                                                          Base-class for abelian symmetry groups.
                                                                          )pydoc");

    cls.def(py::init<Sector, std::string, float64, std::optional<std::string>, bool>(),
            py::arg("trivial_sector"),
            py::arg("group_name"),
            py::arg("num_sectors"),
            py::arg("descriptive_name") = py::none(),
            py::arg("trivial_shift") = true);

    // Abelian defaults (subclasses may override via trampoline).
    cls.def("sector_str", &AbelianGroup::sector_str, py::arg("a"))
      .def("sector_dim", &AbelianGroup::sector_dim, py::arg("a"))
      .def(
        "batch_sector_dim",
        [](AbelianGroup const& self, SectorArray const& a) {
            return vector_i64_to_numpy(self.batch_sector_dim(a));
        },
        py::arg("a"))
      .def("_n_symbol", &AbelianGroup::_n_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("_f_symbol",
           &AbelianGroup::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("frobenius_schur", &AbelianGroup::frobenius_schur, py::arg("a"))
      .def("qdim", &AbelianGroup::qdim, py::arg("a"))
      .def("sqrt_qdim", &AbelianGroup::sqrt_qdim, py::arg("a"))
      .def("inv_sqrt_qdim", &AbelianGroup::inv_sqrt_qdim, py::arg("a"))
      .def("_b_symbol", &AbelianGroup::_b_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("_r_symbol", &AbelianGroup::_r_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("_c_symbol",
           &AbelianGroup::_c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("_fusion_tensor",
           &AbelianGroup::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a"),
           py::arg("Z_b"))
      .def("Z_iso", &AbelianGroup::Z_iso, py::arg("a"));
}

} // namespace cyten
