#pragma once

#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/trees.h>
#include <cyten/tools/mappings.h>

#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

struct BraidInstruction
{
    bool codomain = false;
    int64 idx = 0;
    bool overbraid = false;

    bool operator==(BraidInstruction const&) const = default;
};

struct BendInstruction
{
    bool bend_down = false;

    bool operator==(BendInstruction const&) const = default;
};

struct TwistInstruction
{
    bool codomain = false;
    std::vector<int64> idcs;
    bool overtwist = false;

    bool operator==(TwistInstruction const&) const = default;
};

using Instruction = std::variant<BraidInstruction, BendInstruction, TwistInstruction>;

/// Parse a Python list of bound instruction objects into C++ ``Instruction`` values.
[[nodiscard]] std::vector<Instruction> instructions_from_python(py::object instructions);

/// Convert linear combinations from fusion-tree operations to sparse mapping rows.
[[nodiscard]] SparseMappingFusionTree::Inner to_inner(FusionTreeLinearCombination const& lc);

[[nodiscard]] SparseMappingFusionTreePair::Inner to_inner_pair(
  FusionTreePairLinearCombination const& lc);

using FusionTreeMappingVariant = std::variant<IdentityMappingFusionTree, SparseMappingFusionTree>;

/// Symbolic representation of a map on tensors, defined by the action on tree pairs.
class TensorMapping
{
  public:
    bool is_real;

    explicit TensorMapping(bool is_real_)
      : is_real(is_real_)
    {
    }

    virtual ~TensorMapping() = default;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_instruction(
      Instruction const& instruction,
      bool instruction_is_real,
      std::optional<float64> prune_tol = 1e-15) const;

    [[nodiscard]] virtual std::unique_ptr<TensorMapping> pre_compose_bend_instruction(
      BendInstruction const& instruction,
      bool instruction_is_real) const = 0;

    [[nodiscard]] virtual std::unique_ptr<TensorMapping> pre_compose_braid_instruction(
      BraidInstruction const& instruction,
      bool instruction_is_real) const = 0;

    [[nodiscard]] virtual std::unique_ptr<TensorMapping> pre_compose_twist_instruction(
      TwistInstruction const& instruction,
      bool instruction_is_real) const = 0;

    virtual void prune(float64 tol = 1e-15) = 0;

    [[nodiscard]] virtual FusionTreeData::Ptr transform_tensor(
      FusionTreeData const& data,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_codomain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& codomain_idcs,
      std::vector<int64> const& domain_idcs,
      std::shared_ptr<BlockBackend> block_backend) const = 0;
};

class TreePairMapping : public TensorMapping
{
  public:
    SparseMappingFusionTreePair mapping;

    TreePairMapping(SparseMappingFusionTreePair mapping_, bool is_real_);

    [[nodiscard]] static std::unique_ptr<TreePairMapping> from_identity(
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] static std::unique_ptr<TreePairMapping> from_instructions(
      std::vector<Instruction> const& instructions,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_bend_instruction(
      BendInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_braid_instruction(
      BraidInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_twist_instruction(
      TwistInstruction const& instruction,
      bool instruction_is_real) const override;

    void prune(float64 tol = 1e-15) override;

    [[nodiscard]] FusionTreeData::Ptr transform_tensor(
      FusionTreeData const& data,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_codomain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& codomain_idcs,
      std::vector<int64> const& domain_idcs,
      std::shared_ptr<BlockBackend> block_backend) const override;

  private:
    [[nodiscard]] TreePairMapping pre_compose_fusion_tree_mapping(
      SparseMappingFusionTree const& tree_mapping,
      bool instruction_is_real) const;

    [[nodiscard]] TreePairMapping pre_compose_splitting_tree_mapping(
      SparseMappingFusionTree const& tree_mapping,
      bool instruction_is_real) const;
};

class FactorizedTreeMapping : public TensorMapping
{
  public:
    FusionTreeMappingVariant splitting_tree_mapping;
    FusionTreeMappingVariant fusion_tree_mapping;

    FactorizedTreeMapping(FusionTreeMappingVariant splitting_tree_mapping_,
                          FusionTreeMappingVariant fusion_tree_mapping_,
                          bool is_real_);

    [[nodiscard]] static std::unique_ptr<FactorizedTreeMapping> from_identity(
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] static std::unique_ptr<FactorizedTreeMapping> from_instructions(
      std::vector<Instruction> const& instructions,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      py::object block_inds = py::none());

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_bend_instruction(
      BendInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_braid_instruction(
      BraidInstruction const& instruction,
      bool instruction_is_real) const override;

    [[nodiscard]] std::unique_ptr<TensorMapping> pre_compose_twist_instruction(
      TwistInstruction const& instruction,
      bool instruction_is_real) const override;

    void prune(float64 tol = 1e-15) override;

    [[nodiscard]] FusionTreeData::Ptr transform_tensor(
      FusionTreeData const& data,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_codomain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& codomain_idcs,
      std::vector<int64> const& domain_idcs,
      std::shared_ptr<BlockBackend> block_backend) const override;

  private:
    [[nodiscard]] std::pair<BlockBackend::BlockPtr, bool> transform_splitting_trees(
      BlockBackend::BlockPtr const& old_block,
      BlockBackend::BlockPtr const& out,
      Sector coupled,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr new_codomain,
      std::vector<int64> const& tree_block_axes_1,
      std::shared_ptr<BlockBackend> block_backend) const;

    [[nodiscard]] std::pair<BlockBackend::BlockPtr, bool> transform_fusion_trees(
      BlockBackend::BlockPtr const& old_block,
      BlockBackend::BlockPtr const& out,
      Sector coupled,
      TensorProduct::Ptr domain,
      TensorProduct::Ptr new_domain,
      std::vector<int64> const& tree_block_axes_2,
      std::shared_ptr<BlockBackend> block_backend) const;
};

} // namespace cyten
