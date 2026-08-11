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
#include <vector>

namespace cyten {

/// Check if legs are compatible (equal if `expect_equal`, otherwise mutually dual).
///
/// Arguments are Python sequences of :class:`Leg` / :class:`Space` (including
/// :class:`TensorProduct`), since callers pass either single legs or whole co-domains.
void _check_compatible_legs(py::sequence legs1, py::sequence legs2, bool expect_equal = true);

/// Compose `tensor` with a mask, preserving the leg order of `tensor`.
///
/// Tensor args are ``py::object`` so Python and C++ tensor instances both work until
/// the Tensor hierarchy is monkey-patched.
[[nodiscard]] py::object _compose_with_Mask(py::object tensor, py::object mask, int64 leg_idx);

/// Restricted case of :func:`compose` where we assume that both tensors are SymmetricTensor.
///
/// Is used by both compose and tdot.
[[nodiscard]] py::object _compose_SymmetricTensors(
  py::object tensor1,
  py::object tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Convert tensor from abelian backend to FT backend. Return the data.
[[nodiscard]] FusionTreeData::Ptr _convert_abelian_to_FT(py::object tensor,
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
[[nodiscard]] AbelianBackendData::Ptr _convert_FT_to_abelian(py::object tensor,
                                                             AbelianBackend::Ptr backend,
                                                             Dtype dtype,
                                                             std::string device);

/// Common steps to prepare a SymmetricTensor before a decomposition.
///
/// Returns ``(tensor, new_co_domain, combine_codomain, combine_domain)``.
/// ``new_co_domain`` is a one-factor :class:`TensorProduct` (Python type hint says
/// ``ElementarySpace`` but callers use it as a co-domain).
[[nodiscard]] std::tuple<py::object, TensorProduct::Ptr, bool, bool> _decomposition_prepare(
  py::object tensor,
  bool new_leg_dual);

/// Parse labels for two-leg decompositions (QR, LQ, eigh, …).
[[nodiscard]] std::pair<LegLabel, LegLabel> _decomposition_labels(py::object new_labels);

/// Parse label for :func:`svd`.
[[nodiscard]] std::tuple<LegLabel, LegLabel, LegLabel, LegLabel> _svd_new_labels(
  py::object new_labels);

} // namespace cyten
