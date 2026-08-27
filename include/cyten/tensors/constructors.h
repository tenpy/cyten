#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/labels.h>

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// The identity tensor on a given leg.
///
/// Returns a `DiagonalTensor` if `diagonal` is true, otherwise a `SymmetricTensor`.
///
/// @param leg The space for the identity.
/// @param backend Backend; `nullptr` selects the default.
/// @param labels Optional leg labels; `nullopt` means unlabeled.
/// @param dtype Element dtype.
/// @param device Optional device string; `nullopt` uses the backend default.
/// @param diagonal If true, return a diagonal identity.
[[nodiscard]] TensorPtr eye(Space::Ptr leg,
                            TensorBackend::Ptr backend = nullptr,
                            std::optional<LegLabels> labels = std::nullopt,
                            Dtype dtype = Dtype::Float64,
                            std::optional<std::string> device = std::nullopt,
                            bool diagonal = true);

/// Convert an existing tensor, checking (co)domain / backend compatibility.
///
/// @param obj Source tensor.
/// @param codomain,domain Codomain and optional domain.
/// @param backend Optional backend override; `nullptr` keeps `obj`'s backend.
/// @param labels Optional leg labels; `nullopt` keeps existing labels.
/// @param dtype Optional dtype conversion; `nullopt` keeps `obj`'s dtype.
/// @param device Optional device; `nullopt` keeps `obj`'s device.
[[nodiscard]] TensorPtr tensor(TensorCPtr obj,
                               TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain = nullptr,
                               TensorBackend::Ptr backend = nullptr,
                               std::optional<LegLabels> labels = std::nullopt,
                               std::optional<Dtype> dtype = std::nullopt,
                               std::optional<std::string> device = std::nullopt);

/// Convert a dense block to a `SymmetricTensor`.
///
/// @param obj Dense block data.
/// @param codomain,domain Codomain and optional domain.
/// @param backend Backend; `nullptr` selects the default.
/// @param labels Optional leg labels; `nullopt` means unlabeled.
/// @param dtype Optional dtype; `nullopt` is inferred from `obj`.
/// @param device Optional device; `nullopt` uses the backend default.
/// @param understood_braiding Whether braiding conventions are already applied.
[[nodiscard]] SymmetricTensorPtr tensor(BlockBackend::BlockPtr obj,
                                        TensorProduct::Ptr codomain,
                                        TensorProduct::Ptr domain = nullptr,
                                        TensorBackend::Ptr backend = nullptr,
                                        std::optional<LegLabels> labels = std::nullopt,
                                        std::optional<Dtype> dtype = std::nullopt,
                                        std::optional<std::string> device = std::nullopt,
                                        bool understood_braiding = false);

/// Add a trivial leg to a tensor.
///
/// A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.
/// `DiagonalTensor` and `Mask` do not support adding legs and are converted to
/// `SymmetricTensor` first.
///
/// The position of the new leg can be specified in three mutually exclusive ways via
/// `legs_pos`, `codomain_pos`, or `domain_pos`. If `legs_pos` is used,
/// `result.legs[legs_pos]` will be the trivial leg. In most cases that unambiguously
/// assigns it to either the domain or the codomain. If ambiguous
/// (`legs_pos == num_codomain_legs`), it is added to the codomain. Alternatively, it can
/// be added to the codomain at `codomain[codomain_pos]` or to the domain at `domain_pos`.
/// Note the implications for `is_dual`: with `legs_pos`,
/// `result.legs[legs_pos].is_dual == is_dual`, but with `domain_pos`,
/// `result.domain[domain_pos].is_dual == is_dual` (mutually opposite conventions).
/// Per default we use `legs_pos == 0`, i.e. add at `legs[0]` / `codomain[0]`.
///
/// @param tens The tensor to add a leg to.
/// @param legs_pos Position in `legs`; `nullopt` if unused.
/// @param codomain_pos Position in the codomain; `nullopt` if unused.
/// @param domain_pos Position in the domain; `nullopt` if unused.
/// @param label Label for the new leg; `nullopt` for unlabeled.
/// @param is_dual If true, add a dual (bra-like) leg.
[[nodiscard]] TensorPtr add_trivial_leg(TensorCPtr tens,
                                        std::optional<int64> legs_pos = std::nullopt,
                                        std::optional<int64> codomain_pos = std::nullopt,
                                        std::optional<int64> domain_pos = std::nullopt,
                                        LegLabel label = std::nullopt,
                                        bool is_dual = false);

/// Return a zero tensor with the same type, dtype, legs, backend and labels.
///
/// @param tensor Template tensor whose metadata is copied.
[[nodiscard]] TensorPtr zero_like(TensorCPtr tensor);

/// Stack a grid of tensors along existing legs.
///
/// Null cells are interpreted as all-zero tensors.
/// The tensors are stacked along the first leg in their codomain and the final leg in their
/// domain. The resulting legs are
/// @f$result.codomain[0] = V = \bigoplus_m V_m@f$ and
/// @f$result.domain[-1] = W = \bigoplus_n W_n@f$, where @f$V_m@f$ is the first codomain leg
/// of all tensors in the @f$m@f$-th row `grid[m]`, and @f$W_n@f$ is the last domain leg of all
/// tensors in the @f$n@f$-th column, i.e. for the tensors `[row[n] for row in grid]`.
///
/// Graphically::
///
///     |                                                      W
///     |                                              │   │ ┏━┷━┓
///     |                                              │   │ ┃p_n┃
///     |                  W                           │   │ ┗━┯━┛
///     |          │   │   │                           │   │   │ W_n
///     |       ┏━━┷━━━┷━━━┷━━┓                     ┏━━┷━━━┷━━━┷━━┓
///     |       ┃     res     ┃    ==   sum_{m,n}   ┃  grid[m][n] ┃
///     |       ┗┯━━━┯━━━┯━━━┯┛                     ┗┯━━━┯━━━┯━━━┯┛
///     |        │   │   │   │                   V_m │   │   │   │
///     |        V                                 ┏━┷━┓ │   │   │
///     |                                          ┃i_m┃ │   │   │
///     |                                          ┗━┯━┛ │   │   │
///     |
///
///
/// where @f$p_n : W = \bigoplus_{n'} W_{n'} \to W_n@f$ is the projection map of the direct sum
/// and @f$i_m : V_m \to \bigoplus_{m'} V_{m'}@f$ the inclusion.
///
/// @param grid Row-major grid of tensors (null cell = zero). All legs except those along
///     which stacking happens must match across the grid; tensors in the same row share the
///     first codomain leg, and tensors in the same column share the last domain leg.
/// @param labels Optional labels for the result; `nullopt` means unlabeled.
/// @param dtype Optional dtype; `nullopt` uses the common dtype of the grid.
[[nodiscard]] TensorPtr tensor_from_grid(std::vector<std::vector<TensorPtr>> grid,
                                         std::optional<LegLabels> labels = std::nullopt,
                                         std::optional<Dtype> dtype = std::nullopt);

/// Projection Mask onto summand ``i`` of a `DirectSumSpace`.
///
/// The large leg is `space`; the small leg is isomorphic to ``space.spaces[i]``
/// (built from the kept multiplicities). Negative ``i`` indexes from the end.
[[nodiscard]] MaskPtr projection_onto_summand(DirectSumSpace::CPtr space,
                                              int64 i,
                                              TensorBackend::Ptr backend = nullptr,
                                              std::optional<LegLabels> labels = std::nullopt,
                                              std::optional<std::string> device = std::nullopt);

/// Inclusion Mask of summand ``i`` into a `DirectSumSpace` (dagger of the projection).
[[nodiscard]] MaskPtr inclusion_of_summand(DirectSumSpace::CPtr space,
                                           int64 i,
                                           TensorBackend::Ptr backend = nullptr,
                                           std::optional<LegLabels> labels = std::nullopt,
                                           std::optional<std::string> device = std::nullopt);

/// Unit vector selecting summand ``i`` of a `DirectSumSpace`.
///
/// Requires ``space.spaces[i]`` to be the one-dimensional trivial sector.
/// Returns a rank-1 `SymmetricTensor` (codomain = ``space``, empty domain),
/// obtained by converting the inclusion Mask and squeezing the trivial domain
/// leg. Optional `labels` must have length 1 (the remaining leg).
[[nodiscard]] SymmetricTensorPtr unit_vector_of_summand(
  DirectSumSpace::CPtr space,
  int64 i,
  TensorBackend::Ptr backend = nullptr,
  std::optional<LegLabels> labels = std::nullopt,
  std::optional<Dtype> dtype = std::nullopt,
  std::optional<std::string> device = std::nullopt);

} // namespace cyten
