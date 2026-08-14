#pragma once

#include <cyten/cyten.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/ops_algebra.h>

#include <optional>
#include <variant>
#include <vector>

namespace cyten {

/// Single bool (broadcast) or per-leg ``bool | None``.
using BendRight = std::variant<bool, std::vector<std::optional<bool>>>;
/// Single bool (broadcast) or one flag per combined group.
using PipeDualities = std::variant<bool, std::vector<bool>>;

/// Move legs between codomain and domain without changing the order of ``tensor.legs``.
[[nodiscard]] TensorPtr bend_legs(TensorCPtr tensor,
                                  std::optional<int64> num_codomain_legs = std::nullopt,
                                  std::optional<int64> num_domain_legs = std::nullopt);

/// Check if two tensors have the same legs.
void check_same_legs(TensorCPtr t1, TensorCPtr t2);

/// Combine (multiple) groups of legs, each to a :class:`LegPipe`.
[[nodiscard]] TensorPtr combine_legs(TensorCPtr tensor,
                                     std::vector<std::vector<LegRef>> which_legs,
                                     std::optional<PipeDualities> pipe_dualities = std::nullopt,
                                     std::optional<std::vector<Leg::Ptr>> pipes = std::nullopt,
                                     std::optional<LevelsSpec> levels = std::nullopt);

/// Combine legs of a tensor into two combined LegPipes (matrix form).
[[nodiscard]] TensorPtr combine_to_matrix(
  TensorCPtr tensor,
  std::optional<std::vector<LegRef>> codomain = std::nullopt,
  std::optional<std::vector<LegRef>> domain = std::nullopt,
  std::optional<LevelsSpec> levels = std::nullopt);

/// Move one leg of a tensor to a specified position.
[[nodiscard]] TensorPtr move_leg(TensorCPtr tensor,
                                 LegRef which_leg,
                                 std::optional<int64> codomain_pos = std::nullopt,
                                 std::optional<int64> domain_pos = std::nullopt,
                                 std::optional<LevelsSpec> levels = std::nullopt,
                                 std::optional<BendRight> bend_right = std::nullopt);

/// Permute the legs of a tensor by braiding legs and bending lines.
[[nodiscard]] TensorPtr permute_legs(TensorCPtr tensor,
                                     std::optional<std::vector<LegRef>> codomain = std::nullopt,
                                     std::optional<std::vector<LegRef>> domain = std::nullopt,
                                     std::optional<LevelsSpec> levels = std::nullopt,
                                     std::optional<BendRight> bend_right = std::nullopt);

/// Split legs that were previously combined using :func:`combine_legs`.
[[nodiscard]] TensorPtr split_legs(TensorCPtr tensor,
                                   std::optional<std::vector<LegRef>> legs = std::nullopt);

/// Remove trivial legs.
[[nodiscard]] TensorPtr squeeze_legs(TensorCPtr tensor,
                                     std::optional<std::vector<LegRef>> legs = std::nullopt);

/// Contract one multiplicity copy of one sector on `leg`.
///
/// The leftover one-sector space becomes the charge leg of a :class:`ChargedTensor`.
/// The public-index form requires :attr:`Symmetry.can_be_dropped`.
[[nodiscard]] ChargedTensorPtr slice_leg(TensorCPtr tensor, LegRef leg, int64 idx);
[[nodiscard]] ChargedTensorPtr slice_leg(TensorCPtr tensor,
                                         LegRef leg,
                                         Sector const& sector,
                                         int64 multiplicity = 0);

} // namespace cyten
