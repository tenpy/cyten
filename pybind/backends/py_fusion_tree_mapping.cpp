#include <cyten/backends/fusion_tree_mapping.h>
#include <cyten/backends/fusion_tree_permute.h>

#include "../py_cyten_pybind11.h"

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

void
bind_fusion_tree_mapping(py::module_& m)
{
    py::class_<BraidInstruction>(m, "BraidInstruction")
      .def(py::init<bool, int64, bool>(),
           py::arg("codomain"),
           py::arg("idx"),
           py::arg("overbraid"))
      .def_readwrite("codomain", &BraidInstruction::codomain)
      .def_readwrite("idx", &BraidInstruction::idx)
      .def_readwrite("overbraid", &BraidInstruction::overbraid)
      .def(py::self == py::self)
      .def("__repr__", [](BraidInstruction const& i) {
          return std::format("BraidInstruction(codomain={}, idx={}, overbraid={})",
                             i.codomain,
                             i.idx,
                             i.overbraid);
      });

    py::class_<BendInstruction>(m, "BendInstruction")
      .def(py::init<bool>(), py::arg("bend_down"))
      .def_readwrite("bend_down", &BendInstruction::bend_down)
      .def(py::self == py::self)
      .def("__repr__", [](BendInstruction const& i) {
          return std::format("BendInstruction(bend_down={})", i.bend_down);
      });

    py::class_<TwistInstruction>(m, "TwistInstruction")
      .def(py::init<bool, std::vector<int64>, bool>(),
           py::arg("codomain"),
           py::arg("idcs"),
           py::arg("overtwist"))
      .def_readwrite("codomain", &TwistInstruction::codomain)
      .def_readwrite("idcs", &TwistInstruction::idcs)
      .def_readwrite("overtwist", &TwistInstruction::overtwist)
      .def(py::self == py::self)
      .def("__repr__", [](TwistInstruction const& i) {
          std::string idcs_str = "[";
          for (std::size_t k = 0; k < i.idcs.size(); ++k) {
              if (k > 0) {
                  idcs_str += ", ";
              }
              idcs_str += std::to_string(i.idcs[k]);
          }
          idcs_str += "]";
          return std::format("TwistInstruction(codomain={}, idcs={}, overtwist={})",
                             i.codomain,
                             idcs_str,
                             i.overtwist);
      });

    py::class_<TreePairMapping> tree_pair(m, "TreePairMapping");
    tree_pair.def(py::init<SparseMappingFusionTreePair, bool>(),
                  py::arg("mapping"),
                  py::arg("is_real"))
      .def_readwrite("is_real", &TreePairMapping::is_real)
      .def_readwrite("mapping", &TreePairMapping::mapping)
      .def_static("from_identity",
                  &TreePairMapping::from_identity,
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none())
      .def_static("from_instructions",
                  &TreePairMapping::from_instructions,
                  py::arg("instructions"),
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none())
      .def("prune", &TreePairMapping::prune, py::arg("tol") = 1e-15)
      .def(
        "transform_tensor",
        [](TreePairMapping const& self,
           FusionTreeData const& data,
           TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain,
           TensorProduct::Ptr new_codomain,
           TensorProduct::Ptr new_domain,
           std::vector<int64> codomain_idcs,
           std::vector<int64> domain_idcs,
           py::object block_backend) {
            auto backend = block_backend.cast<std::shared_ptr<BlockBackend>>();
            return self.transform_tensor(data,
                                         codomain,
                                         domain,
                                         new_codomain,
                                         new_domain,
                                         codomain_idcs,
                                         domain_idcs,
                                         backend);
        },
        py::arg("data"),
        py::arg("codomain"),
        py::arg("domain"),
        py::arg("new_codomain"),
        py::arg("new_domain"),
        py::arg("codomain_idcs"),
        py::arg("domain_idcs"),
        py::arg("block_backend"));

    py::class_<FactorizedTreeMapping> fact(m, "FactorizedTreeMapping");
    fact.def(py::init<FusionTreeMappingVariant, FusionTreeMappingVariant, bool>(),
             py::arg("splitting_tree_mapping"),
             py::arg("fusion_tree_mapping"),
             py::arg("is_real"))
      .def_readwrite("is_real", &FactorizedTreeMapping::is_real)
      .def_readwrite("splitting_tree_mapping", &FactorizedTreeMapping::splitting_tree_mapping)
      .def_readwrite("fusion_tree_mapping", &FactorizedTreeMapping::fusion_tree_mapping)
      .def_static("from_identity",
                  &FactorizedTreeMapping::from_identity,
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none())
      .def_static("from_instructions",
                  &FactorizedTreeMapping::from_instructions,
                  py::arg("instructions"),
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none())
      .def("prune", &FactorizedTreeMapping::prune, py::arg("tol") = 1e-15)
      .def(
        "transform_tensor",
        [](FactorizedTreeMapping const& self,
           FusionTreeData const& data,
           TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain,
           TensorProduct::Ptr new_codomain,
           TensorProduct::Ptr new_domain,
           std::vector<int64> codomain_idcs,
           std::vector<int64> domain_idcs,
           py::object block_backend) {
            auto backend = block_backend.cast<std::shared_ptr<BlockBackend>>();
            return self.transform_tensor(data,
                                         codomain,
                                         domain,
                                         new_codomain,
                                         new_domain,
                                         codomain_idcs,
                                         domain_idcs,
                                         backend);
        },
        py::arg("data"),
        py::arg("codomain"),
        py::arg("domain"),
        py::arg("new_codomain"),
        py::arg("new_domain"),
        py::arg("codomain_idcs"),
        py::arg("domain_idcs"),
        py::arg("block_backend"));

    py::class_<PermuteLegsInstructionEngine>(m, "PermuteLegsInstructionEngine")
      .def(py::init<int64,
                    int64,
                    std::vector<int64>,
                    std::vector<int64>,
                    std::vector<std::optional<int64>>,
                    std::vector<std::optional<bool>>,
                    bool>(),
           py::arg("num_codomain_legs"),
           py::arg("num_domain_legs"),
           py::arg("codomain_idcs"),
           py::arg("domain_idcs"),
           py::arg("levels"),
           py::arg("bend_right"),
           py::arg("has_symmetric_braid"))
      .def_readwrite("num_legs", &PermuteLegsInstructionEngine::num_legs)
      .def_readwrite("has_symmetric_braid", &PermuteLegsInstructionEngine::has_symmetric_braid)
      .def_readwrite("num_codomain_legs", &PermuteLegsInstructionEngine::num_codomain_legs)
      .def_readwrite("num_domain_legs", &PermuteLegsInstructionEngine::num_domain_legs)
      .def_readwrite("target_positions", &PermuteLegsInstructionEngine::target_positions)
      .def_readwrite("levels", &PermuteLegsInstructionEngine::levels)
      .def(
        "evaluate_instructions",
        [](PermuteLegsInstructionEngine& self) {
            auto inst = self.evaluate_instructions();
            py::list out;
            for (Instruction const& i : inst) {
                std::visit([&](auto const& x) { out.append(py::cast(x)); }, i);
            }
            return out;
        })
      .def("verify",
           &PermuteLegsInstructionEngine::verify,
           py::arg("num_codomain_legs"),
           py::arg("num_domain_legs"),
           py::arg("codomain_idcs"),
           py::arg("domain_idcs"));
}

} // namespace cyten
