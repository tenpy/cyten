#pragma once

#include <cyten/backends/fusion_tree_mapping.h>

#include <optional>
#include <vector>

namespace cyten {

enum class ShouldBend
{
    None,
    Left,
    Right
};

/// Helper class to build the basic instructions that realize a leg permutation.
///
/// The strategy is to have a stateful instance of this class that represents a list
/// of `instructions` that have already been deduced, as well as attributes that encode
/// what needs to be done still.
///
/// Typical usage is to call `evaluate_instructions` once and consider the rest of the
/// methods as internals.
class PermuteLegsInstructionEngine
{
  public:
    int64 num_legs = 0;
    bool has_symmetric_braid = false;

    int64 num_codomain_legs = 0;
    int64 num_domain_legs = 0;
    std::vector<std::optional<int64>> target_positions;
    std::vector<ShouldBend> should_bend;
    std::vector<std::optional<int64>> levels;
    std::vector<Instruction> instructions;

    PermuteLegsInstructionEngine(int64 num_codomain_legs_,
                                 int64 num_domain_legs_,
                                 std::vector<int64> codomain_idcs,
                                 std::vector<int64> domain_idcs,
                                 std::vector<std::optional<int64>> levels_,
                                 std::vector<std::optional<bool>> bend_right,
                                 bool has_symmetric_braid_);

    [[nodiscard]] std::vector<Instruction> evaluate_instructions();

    /// Verify that the `instructions` reproduce the target leg permutation.
    ///
    /// Note: we only check if the legs end up where they are supposed to, we do not verify
    /// braid chiralities.
    /// TODO should we?
    ///
    /// @param num_codomain_legs, num_domain_legs The leg numbers of the original non-permuted tensor.
    /// @param codomain_idcs, domain_idcs The target permutations.
    void verify(int64 num_codomain_legs_,
                int64 num_domain_legs_,
                std::vector<int64> const& codomain_idcs,
                std::vector<int64> const& domain_idcs) const;

  private:
    [[nodiscard]] bool compare_levels(int64 idx_1, int64 idx_2) const;

    [[nodiscard]] std::pair<int64, int64> do_initial_codomain_permutation();
    void do_codomain_bends(int64 num_left_bends, int64 num_right_bends);
    [[nodiscard]] std::pair<int64, int64> do_domain_permutation();
    void do_domain_bends(int64 num_left_bends, int64 num_right_bends);
    void do_final_codomain_permutation();

    void bend(bool bend_down);
    void move_leg(int64 start, int64 goal, std::optional<bool> over = std::nullopt);
    void swap(int64 idx, std::optional<bool> over = std::nullopt);
};

} // namespace cyten
