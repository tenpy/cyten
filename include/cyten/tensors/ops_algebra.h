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
/// Per-leg braid levels; ``nullopt`` entries are unspecified.
using LevelsSpec = std::vector<std::optional<int64>>;

/// Checks if two tensors are equal up to numerical tolerance.
[[nodiscard]] bool almost_equal(TensorCPtr tensor_1,
                                TensorCPtr tensor_2,
                                float64 rtol = 1e-5,
                                float64 atol = 1e-8,
                                bool allow_different_types = false);

/// Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.
[[nodiscard]] TensorPtr apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.
[[nodiscard]] TensorPtr enlarge_leg(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// The hermitian conjugate tensor, a.k.a the dagger of a tensor.
[[nodiscard]] TensorPtr dagger(TensorCPtr tensor);

/// Tensor contraction as map composition. Requires ``tensor1.domain == tensor2.codomain``.
///
/// If both tensors have no remaining open legs, returns a scalar.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// If the given tensors have the same device, return it. Raise otherwise.
[[nodiscard]] std::string get_same_device(std::vector<TensorCPtr> const& tensors,
                                          std::string const& error_msg = "Incompatible devices.");

/// The Frobenius inner product of two tensors.
[[nodiscard]] BlockBackend::Scalar inner(TensorCPtr A, TensorCPtr B, bool do_dagger = true);

/// Inner product of two :class:`VectorLike` objects (Tensor or DirectSum).
[[nodiscard]] BlockBackend::Scalar inner(VectorLikeCPtr A,
                                         VectorLikeCPtr B,
                                         bool do_dagger = true);

/// If the tensor is a scalar (single-sector, all multiplicities 1).
[[nodiscard]] bool is_scalar(TensorCPtr obj);

/// If the tensor is a scalar (with only trivial legs), convert to a Scalar.
[[nodiscard]] BlockBackend::Scalar item(TensorCPtr tensor);

/// The linear combination ``a * v + b * w``.
[[nodiscard]] TensorPtr linear_combination(BlockBackend::Scalar const& a,
                                           TensorCPtr v,
                                           BlockBackend::Scalar const& b,
                                           TensorCPtr w);

/// Linear combination of two :class:`VectorLike` objects.
[[nodiscard]] VectorLikePtr linear_combination(BlockBackend::Scalar const& a,
                                               VectorLikeCPtr v,
                                               BlockBackend::Scalar const& b,
                                               VectorLikeCPtr w);

/// The Frobenius norm of a Tensor.
[[nodiscard]] BlockBackend::Scalar norm(TensorCPtr tensor);

/// Norm of a :class:`VectorLike`.
[[nodiscard]] BlockBackend::Scalar norm(VectorLikeCPtr vec);

/// An equivalent tensor (with the same entries) on another device.
[[nodiscard]] TensorPtr on_device(TensorCPtr tensor, std::string device, bool copy = true);

/// The outer product, or tensor product.
[[nodiscard]] TensorPtr outer(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Tensor contraction / composition involving only a part of the full (co)domain.
[[nodiscard]] TensorPtr partial_compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  LegRef tensor1_first_leg,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform a partial trace over pairs of legs.
///
/// If all legs are traced, returns a scalar.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> partial_trace(
  TensorCPtr tensor,
  std::vector<std::vector<LegRef>> pairs,
  std::optional<LevelsSpec> levels = std::nullopt);

/// The Moore-Penrose pseudo-inverse of a tensor.
[[nodiscard]] TensorPtr pinv(TensorCPtr tensor, float64 cutoff = 1e-15);

/// The scalar multiplication ``a * v``.
[[nodiscard]] TensorPtr scalar_multiply(BlockBackend::Scalar const& a, TensorCPtr v);

/// Scalar multiplication of a :class:`VectorLike`.
[[nodiscard]] VectorLikePtr scalar_multiply(BlockBackend::Scalar const& a, VectorLikeCPtr v);

/// Contract one `leg` of `tensor` with a diagonal tensor.
[[nodiscard]] TensorPtr scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg);

/// General tensor contraction, connecting arbitrary pairs of (matching!) legs.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> tdot(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::vector<LegRef> legs1,
  std::vector<LegRef> legs2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform the full trace. Requires ``tensor.domain == tensor.codomain``.
[[nodiscard]] BlockBackend::Scalar trace(TensorCPtr tensor);

/// The transpose of a tensor.
[[nodiscard]] TensorPtr transpose(TensorCPtr tensor);

} // namespace cyten
