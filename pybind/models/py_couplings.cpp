#include <cyten/models/couplings.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"
#include "docstrings/models/couplings.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

namespace {

py::object
not_implemented()
{
    return py::module_::import("builtins").attr("NotImplemented");
}

} // namespace

void
bind_models_couplings(py::module_& m)
{
    m.def("freeze", &freeze, py::arg("obj"), DOC(cyten, freeze));

    m.def("_adjacent_transpositions",
          &adjacent_transpositions,
          py::arg("permutation"),
          doc_cpp_ref(R"pydoc(_adjacent_transpositions)pydoc", "cyten::_adjacent_transpositions"));

    m.def("space_to_dict", &space_to_dict, py::arg("space"));

    py::class_<Coupling, py::smart_holder> coupling(m, "Coupling");
    coupling.doc() = DOC(cyten, Coupling);

    coupling
      .def(py::init<std::vector<Site::Ptr>,
                    std::vector<SymmetricTensorPtr>,
                    std::optional<std::string>>(),
           py::arg("sites"),
           py::arg("factorization"),
           py::arg("name") = py::none())
      .def_readwrite("sites", &Coupling::sites)
      .def_readwrite("factorization", &Coupling::factorization)
      .def_readwrite("name", &Coupling::name)
      .def_readwrite("_levels", &Coupling::_levels)
      .def_property_readonly("_permuted",
                             [](Coupling const& self) {
                                 py::list out;
                                 for (auto const& [perm, obj] : self._permuted_py) {
                                     out.append(py::make_tuple(py::cast(perm), obj));
                                 }
                                 return out;
                             })
      .def_static("from_dense_block",
                  &Coupling::from_dense_block,
                  py::arg("operator"),
                  py::arg("sites"),
                  py::arg("name") = py::none(),
                  py::arg("dtype") = py::none(),
                  py::arg("understood_braiding") = false,
                  py::arg("cutoff_singular_values") = py::none(),
                  DOC(cyten, Coupling, from_dense_block))
      .def_static("from_tensor",
                  &Coupling::from_tensor,
                  py::arg("operator"),
                  py::arg("sites"),
                  py::arg("name") = py::none(),
                  py::arg("cutoff") = py::none(),
                  DOC(cyten, Coupling, from_tensor))
      .def("to_tensor", &Coupling::to_tensor, DOC(cyten, Coupling, to_tensor))
      .def("to_numpy",
           &Coupling::to_numpy,
           py::arg("leg_order") = py::none(),
           py::arg("dtype") = py::none(),
           py::arg("understood_braiding") = false,
           DOC(cyten, Coupling, to_numpy))
      .def("stretch_with_identities",
           &Coupling::stretch_with_identities,
           py::arg("all_sites"),
           py::arg("coupling_positions"),
           DOC(cyten, Coupling, stretch_with_identities))
      .def(
        "permute",
        [](Coupling& self,
           std::vector<int64> const& permutation,
           py::object levels,
           py::object over_braid) -> py::object {
            for (auto const& [key, obj] : self._permuted_py) {
                if (key == permutation) {
                    return obj;
                }
            }
            std::optional<LevelsSpec> levels_spec;
            if (!levels.is_none()) {
                levels_spec = levels.cast<LevelsSpec>();
            }
            std::optional<std::vector<std::optional<bool>>> over_braid_spec;
            if (!over_braid.is_none()) {
                over_braid_spec = over_braid.cast<std::vector<std::optional<bool>>>();
            }
            Coupling result = self.permute(permutation, levels_spec, over_braid_spec);
            py::object result_obj = py::cast(std::move(result));
            self._permuted_py.emplace_back(permutation, result_obj);
            return result_obj;
        },
        py::arg("permutation"),
        py::arg("levels") = py::none(),
        py::arg("over_braid") = py::none(),
        DOC(cyten, Coupling, permute))
      .def(
        "_key",
        [](Coupling const& self) { return std::get<0>(self.key()); },
        doc_cpp_ref(R"pydoc(_key)pydoc", "cyten::Coupling::_key()"))
      .def(
        "__eq__",
        [](Coupling const& self, py::handle other) -> py::object {
            if (!py::isinstance<Coupling>(other)) {
                return not_implemented();
            }
            return py::cast(self == other.cast<Coupling>());
        },
        py::arg("other"))
      .def_property_readonly("num_sites", &Coupling::num_sites)
      .def("__hash__", &Coupling::hash)
      .def("__repr__", &Coupling::repr)
      .def("test_sanity", &Coupling::test_sanity, DOC(cyten, Coupling, test_sanity));

    m.def("spin_spin_coupling",
          &spin_spin_coupling,
          py::arg("sites"),
          py::arg("Jx") = 0,
          py::arg("Jy") = 0,
          py::arg("Jz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(spin_spin_coupling)pydoc", "cyten::Coupling::spin_spin_coupling()"));

    m.def(
      "spin_field_coupling",
      &spin_field_coupling,
      py::arg("sites"),
      py::arg("hx") = 0,
      py::arg("hy") = 0,
      py::arg("hz") = 0,
      py::arg("backend") = py::none(),
      py::arg("device") = py::none(),
      py::arg("name") = py::none(),
      doc_cpp_ref(R"pydoc(spin_field_coupling)pydoc", "cyten::Coupling::spin_field_coupling()"));

    m.def("aklt_coupling",
          &aklt_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(aklt_coupling)pydoc", "cyten::Coupling::aklt_coupling()"));

    m.def(
      "heisenberg_coupling",
      &heisenberg_coupling,
      py::arg("sites"),
      py::arg("J") = 1,
      py::arg("backend") = py::none(),
      py::arg("device") = py::none(),
      py::arg("name") = py::none(),
      doc_cpp_ref(R"pydoc(heisenberg_coupling)pydoc", "cyten::Coupling::heisenberg_coupling()"));

    m.def("chiral_3spin_coupling",
          &chiral_3spin_coupling,
          py::arg("sites"),
          py::arg("chi") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(chiral_3spin_coupling)pydoc",
                      "cyten::Coupling::chiral_3spin_coupling()"));

    m.def("chemical_potential",
          &chemical_potential,
          py::arg("sites"),
          py::arg("mu"),
          py::arg("species") = all_species_sentinel(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(chemical_potential)pydoc", "cyten::Coupling::chemical_potential()"));

    m.def("onsite_interaction",
          &onsite_interaction,
          py::arg("sites"),
          py::arg("U") = 1,
          py::arg("species") = all_species_sentinel(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(onsite_interaction)pydoc", "cyten::Coupling::onsite_interaction()"));

    m.def("density_density_interaction",
          &density_density_interaction,
          py::arg("sites"),
          py::arg("V") = 1,
          py::arg("species_i") = all_species_sentinel(),
          py::arg("species_j") = all_species_sentinel(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(density_density_interaction)pydoc",
                      "cyten::Coupling::density_density_interaction()"));

    m.def("hopping",
          &hopping,
          py::arg("sites"),
          py::arg("t") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(hopping)pydoc", "cyten::Coupling::hopping()"));

    m.def("pairing",
          &pairing,
          py::arg("sites"),
          py::arg("Delta") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(pairing)pydoc", "cyten::Coupling::pairing()"));

    m.def("onsite_pairing",
          &onsite_pairing,
          py::arg("sites"),
          py::arg("Delta") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(onsite_pairing)pydoc", "cyten::Coupling::onsite_pairing()"));

    m.def(
      "clock_clock_coupling",
      &clock_clock_coupling,
      py::arg("sites"),
      py::arg("Jx") = 0,
      py::arg("Jz") = 0,
      py::arg("backend") = py::none(),
      py::arg("device") = py::none(),
      py::arg("name") = py::none(),
      doc_cpp_ref(R"pydoc(clock_clock_coupling)pydoc", "cyten::Coupling::clock_clock_coupling()"));

    m.def(
      "clock_field_coupling",
      &clock_field_coupling,
      py::arg("sites"),
      py::arg("hx") = py::none(),
      py::arg("hz") = py::none(),
      py::arg("backend") = py::none(),
      py::arg("device") = py::none(),
      py::arg("name") = py::none(),
      doc_cpp_ref(R"pydoc(clock_field_coupling)pydoc", "cyten::Coupling::clock_field_coupling()"));

    m.def("sector_projection_coupling",
          &sector_projection_coupling,
          py::arg("sites"),
          py::arg("J"),
          py::arg("sector"),
          py::arg("name"),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          doc_cpp_ref(R"pydoc(sector_projection_coupling)pydoc",
                      "cyten::Coupling::sector_projection_coupling()"));

    m.def("gold_coupling",
          &gold_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          doc_cpp_ref(R"pydoc(gold_coupling)pydoc", "cyten::Coupling::gold_coupling()"));
}

} // namespace cyten
