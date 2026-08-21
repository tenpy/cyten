#pragma once

#include <cyten/block_backend/block_backend.h>
#include <cyten/cyten.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/vector_like.h>

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

using LegRef = std::variant<int64, std::string>;
/// Per-leg braid levels; `nullopt` entries are unspecified.
using LevelsSpec = std::vector<std::optional<int64>>;

/// Checks if two tensors are equal up to numerical tolerance.
///
/// We compare the blocks, i.e. the free parameters of the tensors.
/// The tensors count as almost equal if all block-entries, i.e. all their free
/// parameters individually fulfill `abs(a1 - a2) <= atol + rtol * abs(a1)`.
///
/// @param tensor_1,tensor_2 The tensors to compare.
/// @param rtol,atol Relative and absolute tolerances.
/// @param allow_different_types If true, allow comparing tensors of different
///     concrete types.
[[nodiscard]] bool almost_equal(TensorCPtr tensor_1,
                                TensorCPtr tensor_2,
                                float64 rtol = 1e-5,
                                float64 atol = 1e-8,
                                bool allow_different_types = false);

/// Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.
///
/// @param tensor The tensor to project.
/// @param mask Projection mask on `leg`.
/// @param leg Leg index or label to project.
[[nodiscard]] TensorPtr apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.
///
/// @param tensor The tensor to enlarge.
/// @param mask Inclusion mask on `leg`.
/// @param leg Leg index or label to enlarge.
[[nodiscard]] TensorPtr enlarge_leg(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// The hermitian conjugate tensor, a.k.a the dagger of a tensor.
[[nodiscard]] TensorPtr dagger(TensorCPtr tensor);

/// Tensor contraction as map composition. Requires `tensor1.domain == tensor2.codomain`.
///
/// If both tensors have no remaining open legs, returns a scalar.
///
/// @param tensor1,tensor2 Maps to compose: `tensor1` after `tensor2`.
/// @param relabel1,relabel2 Optional label maps applied before composition.
///     `nullopt` means no relabel.
/// @returns The composed tensor, or a scalar if no open legs remain.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// If the given tensors have the same device, return it. Raise otherwise.
///
/// @param tensors Tensors whose devices must agree.
/// @param error_msg Message used if devices differ.
[[nodiscard]] std::string get_same_device(std::vector<TensorCPtr> const& tensors,
                                          std::string const& error_msg = "Incompatible devices.");

/// The Frobenius inner product of two tensors.
///
/// @param A,B Tensors to take the inner product of.
/// @param do_dagger If true, dagger `A` before contracting.
[[nodiscard]] BlockBackend::Scalar inner(TensorCPtr A, TensorCPtr B, bool do_dagger = true);

/// Inner product of two `VectorLike` objects (Tensor or DirectSum).
///
/// @param A,B Vector-like objects to take the inner product of.
/// @param do_dagger If true, dagger `A` before contracting.
[[nodiscard]] BlockBackend::Scalar inner(VectorLikeCPtr A,
                                         VectorLikeCPtr B,
                                         bool do_dagger = true);

/// If the tensor is a scalar (single-sector, all multiplicities 1).
[[nodiscard]] bool is_scalar(TensorCPtr obj);

/// If the tensor is a scalar (with only trivial legs), convert to a Scalar.
[[nodiscard]] BlockBackend::Scalar item(TensorCPtr tensor);

/// The linear combination `a * v + b * w`.
///
/// @param a,b Scalar coefficients.
/// @param v,w Tensors to combine.
[[nodiscard]] TensorPtr linear_combination(BlockBackend::Scalar const& a,
                                           TensorCPtr v,
                                           BlockBackend::Scalar const& b,
                                           TensorCPtr w);

/// Linear combination of two `VectorLike` objects.
///
/// @param a,b Scalar coefficients.
/// @param v,w Vector-like objects to combine.
[[nodiscard]] VectorLikePtr linear_combination(BlockBackend::Scalar const& a,
                                               VectorLikeCPtr v,
                                               BlockBackend::Scalar const& b,
                                               VectorLikeCPtr w);

/// The Frobenius norm of a Tensor.
[[nodiscard]] BlockBackend::Scalar norm(TensorCPtr tensor);

/// Norm of a `VectorLike` (Tensor or DirectSum).
[[nodiscard]] BlockBackend::Scalar norm(VectorLikeCPtr vec);

/// An equivalent tensor (with the same entries) on another device.
///
/// @param tensor Source tensor.
/// @param device Target device name.
/// @param copy If true, always return a new tensor; otherwise may reuse storage.
[[nodiscard]] TensorPtr on_device(TensorCPtr tensor, std::string device, bool copy = true);

/// The outer product, or tensor product.
///
/// @param tensor1,tensor2 Factors of the outer product.
/// @param relabel1,relabel2 Optional label maps. `nullopt` means no relabel.
[[nodiscard]] TensorPtr outer(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Tensor contraction / composition involving only a part of the full (co)domain.
///
/// @param tensor1,tensor2 Tensors to partially compose.
/// @param tensor1_first_leg First leg of `tensor1` involved in the partial
///     contraction.
/// @param relabel1,relabel2 Optional label maps. `nullopt` means no relabel.
[[nodiscard]] TensorPtr partial_compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  LegRef tensor1_first_leg,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform a partial trace over pairs of legs.
///
/// If all legs are traced, returns a scalar.
///
/// @param tensor Tensor to trace.
/// @param pairs Pairs of legs to contract.
/// @param levels Optional per-leg braid levels; `nullopt` entries are unspecified.
/// @returns Remaining tensor, or a scalar if everything is traced.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> partial_trace(
  TensorCPtr tensor,
  std::vector<std::vector<LegRef>> pairs,
  std::optional<LevelsSpec> levels = std::nullopt);

/// The Moore-Penrose pseudo-inverse of a tensor.
///
/// @param tensor Tensor to invert.
/// @param cutoff Singular values below this threshold are discarded.
[[nodiscard]] TensorPtr pinv(TensorCPtr tensor, float64 cutoff = 1e-15);

/// The scalar multiplication `a * v`.
///
/// @param a Scalar factor.
/// @param v Tensor to scale.
[[nodiscard]] TensorPtr scalar_multiply(BlockBackend::Scalar const& a, TensorCPtr v);

/// Scalar multiplication of a `VectorLike`.
///
/// @param a Scalar factor.
/// @param v Vector-like object to scale.
[[nodiscard]] VectorLikePtr scalar_multiply(BlockBackend::Scalar const& a, VectorLikeCPtr v);

/// Contract one `leg` of `tensor` with a diagonal tensor.
///
/// @param tensor Tensor to scale along one axis.
/// @param diag Diagonal tensor contracted onto `leg`.
/// @param leg Leg index or label to contract.
[[nodiscard]] TensorPtr scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg);

/// General tensor contraction, connecting arbitrary pairs of (matching!) legs.
///
/// @param tensor1,tensor2 Tensors to contract.
/// @param legs1,legs2 Matching legs of `tensor1` and `tensor2` to contract.
/// @param relabel1,relabel2 Optional label maps. `nullopt` means no relabel.
/// @returns Contracted tensor, or a scalar if no open legs remain.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> tdot(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::vector<LegRef> legs1,
  std::vector<LegRef> legs2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform the full trace. Requires `tensor.domain == tensor.codomain`.
[[nodiscard]] BlockBackend::Scalar trace(TensorCPtr tensor);

/// The transpose of a tensor.
[[nodiscard]] TensorPtr transpose(TensorCPtr tensor);

} // namespace cyten
