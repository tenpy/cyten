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
    py::class_<BraidInstruction> braid_cls(m, "BraidInstruction");
    braid_cls.doc() = R"pydoc(
        Instruction to braid two neighboring legs.

        Attributes
        ----------
        codomain : bool
            If the braid is in the codomain, otherwise in the domain.
        idx : int
            Which leg of the (co-)domain braids.
            We braid ``(co)domain[idx]`` with ``(co)domain[idx + 1]``
        overbraid : bool
            Specifies the chirality of the braid. An overbraid is a braid where the leg that goes
            from bottom left to top right is on top, see notes below.

        Notes
        -----
        Examples for over-braids::

            |    │    ╲ ╱    │                      │   │   │   │
            |    │     ╱     │                     ┏┷━━━┷━━━┷━━━┷┓
            |    │    ╱ ╲    │                     ┃             ┃
            |   ┏┷━━━┷━━━┷━━━┷┓                    ┗━━┯━━━┯━━━┯━━┛
            |   ┃             ┃         OR             ╲ ╱    │
            |   ┗━━┯━━━┯━━━┯━━┛                         ╱     │
            |      │   │   │                           ╱ ╲    │

        Examples for under-braids::

            |    │    ╲ ╱    │                      │   │   │   │
            |    │     ╲     │                     ┏┷━━━┷━━━┷━━━┷┓
            |    │    ╱ ╲    │                     ┃             ┃
            |   ┏┷━━━┷━━━┷━━━┷┓                    ┗━━┯━━━┯━━━┯━━┛
            |   ┃             ┃         OR             ╲ ╱    │
            |   ┗━━┯━━━┯━━━┯━━┛                         ╲     │
            |      │   │   │                           ╱ ╲    │
        )pydoc";

    braid_cls
      .def(
        py::init<bool, int64, bool>(), py::arg("codomain"), py::arg("idx"), py::arg("overbraid"))
      .def_readwrite("codomain", &BraidInstruction::codomain)
      .def_readwrite("idx", &BraidInstruction::idx)
      .def_readwrite("overbraid", &BraidInstruction::overbraid)
      .def(py::self == py::self)
      .def("__repr__", [](BraidInstruction const& i) {
          return std::format(
            "BraidInstruction(codomain={}, idx={}, overbraid={})", i.codomain, i.idx, i.overbraid);
      });

    py::class_<BendInstruction> bend_cls(m, "BendInstruction");
    bend_cls.doc() = R"pydoc(
        Instruction to bend the rightmost leg of the codomain down (of the domain up).
        )pydoc";

    bend_cls.def(py::init<bool>(), py::arg("bend_down"))
      .def_readwrite("bend_down", &BendInstruction::bend_down)
      .def(py::self == py::self)
      .def("__repr__", [](BendInstruction const& i) {
          return std::format("BendInstruction(bend_down={})", i.bend_down);
      });

    py::class_<TwistInstruction> twist_cls(m, "TwistInstruction");
    twist_cls.doc() = R"pydoc(
        Instruction to apply a twist on one leg.

        Attributes
        ----------
        codomain : bool
            If the twist is in the codomain, otherwise in the domain.
        idcs : list of int
            Which legs of the (co-)domain are twisted; we twist ``(co)domain[idcs]``.
            Must be contiguous.
        overtwist : bool
            Specifies the chirality of the twist. An overtwist (undertwist) has an overbraid
            (underbraid) at the center, and a cup and cap.

        Notes
        -----
        Let us first illustrate how the chirality is given by :attr:`overtwist`.
        For simplicity, we always show ``idcs=[-1]``.
        Example for over-twists::

            |    │   │   │   │   ╭─╮             │   │   │   │
            |    │   │   │    ╲ ╱  │            ┏┷━━━┷━━━┷━━━┷┓
            |    │   │   │     ╱   │            ┃             ┃
            |    │   │   │    ╱ ╲  │            ┗━━┯━━━┯━━━┯━━┛╭─╮
            |   ┏┷━━━┷━━━┷━━━┷┓  ╰─╯               │   │    ╲ ╱  │
            |   ┃             ┃         OR         │   │     ╱   │
            |   ┗━━┯━━━┯━━━┯━━┛                    │   │    ╱ ╲  │
            |      │   │   │                       │   │   │   ╰─╯

        Examples for under-twists::

            |    │   │   │   │   ╭─╮             │   │   │   │
            |    │   │   │    ╲ ╱  │            ┏┷━━━┷━━━┷━━━┷┓
            |    │   │   │     ╲   │            ┃             ┃
            |    │   │   │    ╱ ╲  │            ┗━━┯━━━┯━━━┯━━┛╭─╮
            |   ┏┷━━━┷━━━┷━━━┷┓  ╰─╯               │   │    ╲ ╱  │
            |   ┃             ┃         OR         │   │     ╲   │
            |   ┗━━┯━━━┯━━━┯━━┛                    │   │    ╱ ╲  │
            |      │   │   │                       │   │   │   ╰─╯

        For multiple legs (``len(idcs) > 1``), we twist them together, e.g.::

            |
            |
            |    │   │   │   │   ╭──────╮
            |    │   │    ╲   ╲ ╱       │
            |    │   │     ╲   ╱   ╭─╮  │
            |    │   │      ╲ ╱ ╲ ╱  │  │
            |    │   │       ╱   ╱   │  │
            |    │   │      ╱ ╲ ╱ ╲  │  │
            |    │   │     ╱   ╱   ╰─╯  │
            |    │   │    ╱   ╱ ╲       │
            |   ┏┷━━━┷━━━┷━━━┷┓  ╰──────╯
            |   ┃             ┃
            |   ┗━━┯━━━┯━━━┯━━┛
            |      │   │   │
        )pydoc";

    twist_cls
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
    tree_pair.doc() = R"pydoc(
        A :class:`TensorMapping`, defined at the level of tree-pairs, i.e. the general case.

        We store the component ``f_{JI} = <X_J @ Y_J | f(X_I @ Y_I)>``,
        which represents ``X_I @ Y_I \mapsto f_{JI} X_J @ Y_J`` as ``mapping[I][J] = f_{JI}``.
        In practice, the keys are ``I = (X_I, Y_I)`` tuples of two FusionTrees.
        )pydoc";

    tree_pair
      .def(py::init<SparseMappingFusionTreePair, bool>(), py::arg("mapping"), py::arg("is_real"))
      .def_readwrite("is_real", &TreePairMapping::is_real)
      .def_readwrite("mapping", &TreePairMapping::mapping)
      .def_static("from_identity",
                  &TreePairMapping::from_identity,
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none(),
                  R"pydoc(
                  The identity mapping.

                  Parameters
                  ----------
                  codomain, domain : TensorProduct
                      The codomain and domain that determine the possible fusion and splitting trees.
                  block_inds : 2D array
                      Same format and meaning as the :attr:`FusionTreeData.block_inds`.
                      If given, we only initialize those components ``X_I @ Y_I -> X_I @ Y_I``
                      where the coupled sector of the tree-pair is pointed to by a row in the `block_inds`,
                      i.e. if we have ``coupled == codomain.sector_decomposition[block_inds[some_idx, 0]]``.
                  )pydoc")
      .def_static("from_instructions",
                  &TreePairMapping::from_instructions,
                  py::arg("instructions"),
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none())
      .def("prune",
           &TreePairMapping::prune,
           py::arg("tol") = 1e-15,
           R"pydoc(
           Remove small contributions with ``abs(coefficient) < tol`` in-place.
           )pydoc")
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
        py::arg("block_backend"),
        R"pydoc(
        Transform a tensor by applying the mapping to its tree-pairs. See class docstring.

        Parameters
        ----------
        data : FusionTreeData
            The data of the input tensor.
        codomain, domain : TensorProduct
            The (co)domain of the input tensor.
        new_codomain, new_domain : TensorProduct
            The (co)domain of the output tensor.
        codomain_idcs, domain_idcs : list of int
            The permutations such that ``new_(co)domain[i] = old_legs[(co)domain_idcs[i]]``.
            This permutation acts on the uncoupled multiplicity indices.
        )pydoc");

    py::class_<FactorizedTreeMapping> fact(m, "FactorizedTreeMapping");
    fact.doc() = R"pydoc(
        A :class:`TensorMapping` that factorizes into maps on single trees.

        In particular, the action of the mapping on a tree pair factorizes as::

            f(X @ Y) = g(X) @ h(Y)

        and we store the component ``X \mapsto g_{X2, X} X2`` as
        ``g_{X2, X} = splitting_tree_mapping[X2][X] = <X2 | X>`` and similarly
        ``h_{Y2, Y} = fusion_tree_mapping[Y2][Y] = <Y2 | Y>`` for ``Y \mapsto h_{Y2, Y} Y2``.
        Note that ``g`` contains the coefficients in a linear combination of splitting trees,
        which are conjugated compared to the analogous linear combination of fusion trees.
        )pydoc";

    fact
      .def(py::init<FusionTreeMappingVariant, FusionTreeMappingVariant, bool>(),
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
                  py::arg("block_inds") = py::none(),
                  R"pydoc(
                  The identity mapping.

                  Parameters
                  ----------
                  codomain, domain : TensorProduct
                      The codomain and domain that determine the possible fusion and splitting trees.
                  block_inds : 2D array
                      Same format and meaning as the :attr:`FusionTreeData.block_inds`.
                      If given, we only initialize those components ``X_I @ Y_I -> X_I @ Y_I``
                      where the coupled sector of the tree-pair is pointed to by a row in the `block_inds`,
                      i.e. if we have ``coupled == codomain.sector_decomposition[block_inds[some_idx, 0]]``.
                  )pydoc")
      .def_static("from_instructions",
                  &FactorizedTreeMapping::from_instructions,
                  py::arg("instructions"),
                  py::arg("codomain"),
                  py::arg("domain"),
                  py::arg("block_inds") = py::none())
      .def("prune",
           &FactorizedTreeMapping::prune,
           py::arg("tol") = 1e-15,
           R"pydoc(
           Remove small contributions with ``abs(coefficient) < tol`` in-place.
           )pydoc")
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
        py::arg("block_backend"),
        R"pydoc(
        Transform a tensor by applying the mapping to its tree-pairs. See class docstring.

        Parameters
        ----------
        data : FusionTreeData
            The data of the input tensor.
        codomain, domain : TensorProduct
            The (co)domain of the input tensor.
        new_codomain, new_domain : TensorProduct
            The (co)domain of the output tensor.
        codomain_idcs, domain_idcs : list of int
            The permutations such that ``new_(co)domain[i] = old_legs[(co)domain_idcs[i]]``.
            This permutation acts on the uncoupled multiplicity indices.
        )pydoc");

    py::class_<PermuteLegsInstructionEngine> perm_cls(m, "PermuteLegsInstructionEngine");
    perm_cls.doc() = R"pydoc(
        Helper class to build the basic instructions that realized a leg permutation.

        The strategy is to have a stateful instance of this class that represents a list
        of :attr:`instructions` that have already been deduced, as well as attributes that encode
        what needs to be done still.

        Typical usage is to call :meth:`evaluate_instructions` once and consider the rest of the
        methods as internals.
        )pydoc";

    perm_cls
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
      .def("evaluate_instructions",
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
           py::arg("domain_idcs"),
           R"pydoc(
           Verify that the :attr:`instructions` reproduce the target leg permutation.

           Note: we only check if the legs end up where they are supposed to, we do not verify
           braid chiralities.
           TODO should we?

           Parameters
           ----------
           num_codomain_legs, num_domain_legs
               The leg numbers of the original non-permuted tensor
           codomain_idcs, domain_idcs
               The target permutations.

           Raises
           ------
           AssertionError
               If an instruction can not be applied or if the target permutation is not reproduced.
           )pydoc");
}

} // namespace cyten
