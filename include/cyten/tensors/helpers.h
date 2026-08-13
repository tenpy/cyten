#pragma once

#include <cyten/backends/abelian.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tensors/tensor.h>

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

/// Check if legs are compatible (equal if `expect_equal`, otherwise mutually dual).
///
/// :class:`TensorProduct` (co)domains use the :class:`Space` overload. Single tensor legs
/// (:class:`ElementarySpace` / :class:`LegPipe`) use the :class:`Leg` overload.
void _check_compatible_legs(std::vector<Leg::Ptr> const& legs1,
                            std::vector<Leg::Ptr> const& legs2,
                            bool expect_equal = true);
void _check_compatible_legs(std::vector<Space::Ptr> const& legs1,
                            std::vector<Space::Ptr> const& legs2,
                            bool expect_equal = true);

/// Compose `tensor` with a mask, preserving the leg order of `tensor`.
[[nodiscard]] TensorPtr _compose_with_Mask(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx);

/// Restricted case of :func:`compose` where we assume that both tensors are SymmetricTensor.
///
/// If both tensors have no remaining open legs, returns a scalar (the contraction).
[[nodiscard]] std::variant<SymmetricTensorPtr, BlockBackend::Scalar> _compose_SymmetricTensors(
  SymmetricTensorCPtr tensor1,
  SymmetricTensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Convert tensor from abelian backend to FT backend. Return the data.
[[nodiscard]] FusionTreeData::Ptr _convert_abelian_to_FT(TensorCPtr tensor,
                                                         FusionTreeBackend::Ptr backend,
                                                         Dtype dtype,
                                                         std::string device);

/// Convert tensor from FT backend to abelian backend. Return the data.
///
/// Notes
/// -----
/// - For abelian symmetries, a fusion tree is completely determined by its uncoupled sectors
/// - This means that each forest blocks consists of a single tree block
/// - The blocks of the abelian backend correspond one-to-one to tree blocks in the FT backend,
///   up to reshaping and transposing
/// - All that remains is to make sure we loop over all of them in an efficient manner.
/// - It is convenient to do the outer loops over combinations of uncoupled sectors
///     - This way, we have the abelian block_inds by construction
///     - we need to compute the coupled sectors to check for valid fusion channels anyway,
///       which gives us the FT block inds with one additional lookup
///     - While we jump back-and-forth between different coupled sectors, and thus different FT
///       block while iterating, we know that we visit the tree-blocks within each FT block *in
///       order*, and we can thus keep track of where we are within each FT block easily.
[[nodiscard]] AbelianBackendData::Ptr _convert_FT_to_abelian(TensorCPtr tensor,
                                                             AbelianBackend::Ptr backend,
                                                             Dtype dtype,
                                                             std::string device);

/// Common steps to prepare a SymmetricTensor before a decomposition.
///
/// Returns ``(tensor, new_co_domain, combine_codomain, combine_domain)``.
/// ``new_co_domain`` is a one-factor :class:`TensorProduct` (Python type hint says
/// ``ElementarySpace`` but callers use it as a co-domain).
[[nodiscard]] std::tuple<SymmetricTensorPtr, TensorProduct::Ptr, bool, bool>
_decomposition_prepare(TensorCPtr tensor, bool new_leg_dual);

/// Parse labels for two-leg decompositions (QR, LQ, eigh, …).
[[nodiscard]] std::pair<LegLabel, LegLabel> _decomposition_labels(LegLabels const& new_labels);

/// Parse label for :func:`svd`. ``nullopt`` means all-unlabelled.
[[nodiscard]] std::tuple<LegLabel, LegLabel, LegLabel, LegLabel> _svd_new_labels(
  std::optional<LegLabels> new_labels);

} // namespace cyten
