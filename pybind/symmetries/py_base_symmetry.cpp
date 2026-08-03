#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/base_symmetry.h>

#include <cmath>
#include <vector>

namespace cyten {

void
bind_base_symmetry(py::module_& m)
{
    py::class_<BaseSymmetry, PyBaseSymmetry, py::smart_holder> cls(m,
                                                                   "BaseSymmetry",
                                                                   R"pydoc(
Common method implementations for both :class:`SymmetryFactor` and :class:`Symmetry`.

This contains the fallback implementations of e.g. :meth:`qdim` in terms of F symbols.
)pydoc");

    cls.def(py::init<FusionStyle, BraidingStyle, Sector, float64, bool, bool>(),
            py::arg("fusion_style"),
            py::arg("braiding_style"),
            py::arg("trivial_sector"),
            py::arg("num_sectors"),
            py::arg("has_complex_topological_data"),
            py::arg("trivial_shift"));

    cls.def_readonly("fusion_style", &BaseSymmetry::fusion_style)
      .def_readonly("braiding_style", &BaseSymmetry::braiding_style)
      .def_property_readonly("trivial_sector",
                             [](BaseSymmetry const& self) { return self.trivial_sector; })
      .def_property_readonly("num_sectors",
                             [](BaseSymmetry const& self) -> py::object {
                                 // Match Python ``int | float``: finite counts as int, else
                                 // float('inf').
                                 if (std::isfinite(self.num_sectors)) {
                                     return py::int_(static_cast<long long>(self.num_sectors));
                                 }
                                 return py::float_(self.num_sectors);
                             })
      .def_readonly("sector_ind_len", &BaseSymmetry::sector_ind_len)
      .def_property_readonly("empty_sector_array",
                             [](BaseSymmetry const& self) { return self.empty_sector_array; })
      .def_readonly("has_complex_topological_data", &BaseSymmetry::has_complex_topological_data)
      .def_readonly("trivial_shift", &BaseSymmetry::trivial_shift);

    cls.def_property_readonly("can_be_dropped", &BaseSymmetry::can_be_dropped)
      .def_property_readonly("has_symmetric_braid", &BaseSymmetry::has_symmetric_braid)
      .def_property_readonly("has_trivial_braid", &BaseSymmetry::has_trivial_braid)
      .def_property_readonly("is_abelian", &BaseSymmetry::is_abelian)
      .def_property_readonly("has_unique_fusion", &BaseSymmetry::has_unique_fusion);

    cls.def("dual_sector", &BaseSymmetry::dual_sector, py::arg("a"))
      .def("_n_symbol", &BaseSymmetry::_n_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("_f_symbol",
           &BaseSymmetry::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("_r_symbol", &BaseSymmetry::_r_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("as_Symmetry", &BaseSymmetry::as_Symmetry)
      .def("is_valid_sector", &BaseSymmetry::is_valid_sector, py::arg("a"))
      .def("fusion_outcomes", &BaseSymmetry::fusion_outcomes, py::arg("a"), py::arg("b"))
      .def("_fusion_tensor",
           &BaseSymmetry::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a"),
           py::arg("Z_b"))
      .def("swap_gate", &BaseSymmetry::swap_gate, py::arg("a"), py::arg("b"))
      .def("Z_iso", &BaseSymmetry::Z_iso, py::arg("a"))
      .def("all_sectors", &BaseSymmetry::all_sectors)
      .def("n_symbol", &BaseSymmetry::n_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("f_symbol",
           &BaseSymmetry::f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("b_symbol", &BaseSymmetry::b_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("r_symbol", &BaseSymmetry::r_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("c_symbol",
           &BaseSymmetry::c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("fusion_tensor",
           &BaseSymmetry::fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a") = false,
           py::arg("Z_b") = false)
      .def("are_valid_sectors", &BaseSymmetry::are_valid_sectors, py::arg("sectors"))
      .def("fusion_outcomes_broadcast",
           &BaseSymmetry::fusion_outcomes_broadcast,
           py::arg("a"),
           py::arg("b"))
      .def("multiple_fusion",
           [](BaseSymmetry const& self, py::args args) {
               std::vector<Sector> sectors;
               sectors.reserve(args.size());
               for (auto h : args) {
                   sectors.push_back(h.cast<Sector>());
               }
               return self.multiple_fusion(sectors);
           })
      .def("multiple_fusion_broadcast",
           [](BaseSymmetry const& self, py::args args) {
               std::vector<SectorArray> sectors;
               sectors.reserve(args.size());
               for (auto h : args) {
                   sectors.push_back(h.cast<SectorArray>());
               }
               return self.multiple_fusion_broadcast(sectors);
           })
      .def("_multiple_fusion_broadcast",
           [](BaseSymmetry const& self, py::args args) {
               std::vector<SectorArray> sectors;
               sectors.reserve(args.size());
               for (auto h : args) {
                   sectors.push_back(h.cast<SectorArray>());
               }
               return self._multiple_fusion_broadcast(sectors);
           })
      .def("can_fuse_to", &BaseSymmetry::can_fuse_to, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("sector_dim", &BaseSymmetry::sector_dim, py::arg("a"))
      .def("batch_sector_dim", &BaseSymmetry::batch_sector_dim, py::arg("a"))
      .def("batch_qdim", &BaseSymmetry::batch_qdim, py::arg("a"))
      .def("sector_str", &BaseSymmetry::sector_str, py::arg("a"))
      .def("dual_sectors", &BaseSymmetry::dual_sectors, py::arg("sectors"))
      .def("frobenius_schur", &BaseSymmetry::frobenius_schur, py::arg("a"))
      .def("qdim", &BaseSymmetry::qdim, py::arg("a"))
      .def("sqrt_qdim", &BaseSymmetry::sqrt_qdim, py::arg("a"))
      .def("inv_sqrt_qdim", &BaseSymmetry::inv_sqrt_qdim, py::arg("a"))
      .def("total_qdim", &BaseSymmetry::total_qdim)
      .def("_b_symbol", &BaseSymmetry::_b_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("_c_symbol",
           &BaseSymmetry::_c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("topological_twist", &BaseSymmetry::topological_twist, py::arg("a"))
      .def("s_matrix_element", &BaseSymmetry::s_matrix_element, py::arg("a"), py::arg("b"))
      .def("s_matrix", &BaseSymmetry::s_matrix);
}

} // namespace cyten
