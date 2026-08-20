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
///
/// We compare the blocks, i.e. the free parameters of the tensors.
/// The tensors count as almost equal if all block-entries, i.e. all their free
/// parameters individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
///
/// Parameters
/// ----------
/// tensor_1, tensor_2 : TensorCPtr
///     The tensors to compare.
/// rtol, atol : float64
///     Relative and absolute tolerances.
/// allow_different_types : bool
///     If true, allow comparing tensors of different concrete types.
[[nodiscard]] bool almost_equal(TensorCPtr tensor_1,
                                TensorCPtr tensor_2,
                                float64 rtol = 1e-5,
                                float64 atol = 1e-8,
                                bool allow_different_types = false);

/// Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.
///
/// Parameters
/// ----------
/// tensor : TensorCPtr
///     The tensor to project.
/// mask : MaskCPtr
///     Projection mask on ``leg``.
/// leg : LegRef
///     Leg index or label to project.
[[nodiscard]] TensorPtr apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.
///
/// Parameters
/// ----------
/// tensor : TensorCPtr
///     The tensor to enlarge.
/// mask : MaskCPtr
///     Inclusion mask on ``leg``.
/// leg : LegRef
///     Leg index or label to enlarge.
[[nodiscard]] TensorPtr enlarge_leg(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// The hermitian conjugate tensor, a.k.a the dagger of a tensor.
[[nodiscard]] TensorPtr dagger(TensorCPtr tensor);

/// Tensor contraction as map composition. Requires ``tensor1.domain == tensor2.codomain``.
///
/// If both tensors have no remaining open legs, returns a scalar.
///
/// Parameters
/// ----------
/// tensor1, tensor2 : TensorCPtr
///     Maps to compose: ``tensor1`` after ``tensor2``.
/// relabel1, relabel2 : std::optional<std::map<std::string, std::string>>
///     Optional label maps applied before composition. ``nullopt`` means no relabel.
///
/// Returns
/// -------
/// std::variant<TensorPtr, BlockBackend::Scalar>
///     The composed tensor, or a scalar if no open legs remain.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// If the given tensors have the same device, return it. Raise otherwise.
///
/// Parameters
/// ----------
/// tensors : std::vector<TensorCPtr>
///     Tensors whose devices must agree.
/// error_msg : std::string
///     Message used if devices differ.
[[nodiscard]] std::string get_same_device(std::vector<TensorCPtr> const& tensors,
                                          std::string const& error_msg = "Incompatible devices.");

/// The Frobenius inner product of two tensors.
///
/// Parameters
/// ----------
/// A, B : TensorCPtr
///     Tensors to take the inner product of.
/// do_dagger : bool
///     If true, dagger ``A`` before contracting.
[[nodiscard]] BlockBackend::Scalar inner(TensorCPtr A, TensorCPtr B, bool do_dagger = true);

/// Inner product of two ``VectorLike`` objects (Tensor or DirectSum).
///
/// Parameters
/// ----------
/// A, B : VectorLikeCPtr
///     Vector-like objects to take the inner product of.
/// do_dagger : bool
///     If true, dagger ``A`` before contracting.
[[nodiscard]] BlockBackend::Scalar inner(VectorLikeCPtr A,
                                         VectorLikeCPtr B,
                                         bool do_dagger = true);

/// If the tensor is a scalar (single-sector, all multiplicities 1).
[[nodiscard]] bool is_scalar(TensorCPtr obj);

/// If the tensor is a scalar (with only trivial legs), convert to a Scalar.
[[nodiscard]] BlockBackend::Scalar item(TensorCPtr tensor);

/// The linear combination ``a * v + b * w``.
///
/// Parameters
/// ----------
/// a, b : BlockBackend::Scalar
///     Scalar coefficients.
/// v, w : TensorCPtr
///     Tensors to combine.
[[nodiscard]] TensorPtr linear_combination(BlockBackend::Scalar const& a,
                                           TensorCPtr v,
                                           BlockBackend::Scalar const& b,
                                           TensorCPtr w);

/// Linear combination of two ``VectorLike`` objects.
///
/// Parameters
/// ----------
/// a, b : BlockBackend::Scalar
///     Scalar coefficients.
/// v, w : VectorLikeCPtr
///     Vector-like objects to combine.
[[nodiscard]] VectorLikePtr linear_combination(BlockBackend::Scalar const& a,
                                               VectorLikeCPtr v,
                                               BlockBackend::Scalar const& b,
                                               VectorLikeCPtr w);

/// The Frobenius norm of a Tensor.
[[nodiscard]] BlockBackend::Scalar norm(TensorCPtr tensor);

/// Norm of a ``VectorLike`` (Tensor or DirectSum).
[[nodiscard]] BlockBackend::Scalar norm(VectorLikeCPtr vec);

/// An equivalent tensor (with the same entries) on another device.
///
/// Parameters
/// ----------
/// tensor : TensorCPtr
///     Source tensor.
/// device : std::string
///     Target device name.
/// copy : bool
///     If true, always return a new tensor; otherwise may reuse storage.
[[nodiscard]] TensorPtr on_device(TensorCPtr tensor, std::string device, bool copy = true);

/// The outer product, or tensor product.
///
/// Parameters
/// ----------
/// tensor1, tensor2 : TensorCPtr
///     Factors of the outer product.
/// relabel1, relabel2 : std::optional<std::map<std::string, std::string>>
///     Optional label maps. ``nullopt`` means no relabel.
[[nodiscard]] TensorPtr outer(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Tensor contraction / composition involving only a part of the full (co)domain.
///
/// Parameters
/// ----------
/// tensor1, tensor2 : TensorCPtr
///     Tensors to partially compose.
/// tensor1_first_leg : LegRef
///     First leg of ``tensor1`` involved in the partial contraction.
/// relabel1, relabel2 : std::optional<std::map<std::string, std::string>>
///     Optional label maps. ``nullopt`` means no relabel.
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
/// Parameters
/// ----------
/// tensor : TensorCPtr
///     Tensor to trace.
/// pairs : std::vector<std::vector<LegRef>>
///     Pairs of legs to contract.
/// levels : std::optional<LevelsSpec>
///     Optional per-leg braid levels; ``nullopt`` entries are unspecified.
///
/// Returns
/// -------
/// std::variant<TensorPtr, BlockBackend::Scalar>
///     Remaining tensor, or a scalar if everything is traced.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> partial_trace(
  TensorCPtr tensor,
  std::vector<std::vector<LegRef>> pairs,
  std::optional<LevelsSpec> levels = std::nullopt);

/// The Moore-Penrose pseudo-inverse of a tensor.
///
/// Parameters
/// ----------
/// tensor : TensorCPtr
///     Tensor to invert.
/// cutoff : float64
///     Singular values below this threshold are discarded.
[[nodiscard]] TensorPtr pinv(TensorCPtr tensor, float64 cutoff = 1e-15);

/// The scalar multiplication ``a * v``.
///
/// Parameters
/// ----------
/// a : BlockBackend::Scalar
///     Scalar factor.
/// v : TensorCPtr
///     Tensor to scale.
[[nodiscard]] TensorPtr scalar_multiply(BlockBackend::Scalar const& a, TensorCPtr v);

/// Scalar multiplication of a ``VectorLike``.
///
/// Parameters
/// ----------
/// a : BlockBackend::Scalar
///     Scalar factor.
/// v : VectorLikeCPtr
///     Vector-like object to scale.
[[nodiscard]] VectorLikePtr scalar_multiply(BlockBackend::Scalar const& a, VectorLikeCPtr v);

/// Contract one `leg` of `tensor` with a diagonal tensor.
///
/// Parameters
/// ----------
/// tensor : TensorCPtr
///     Tensor to scale along one axis.
/// diag : DiagonalTensorCPtr
///     Diagonal tensor contracted onto ``leg``.
/// leg : LegRef
///     Leg index or label to contract.
[[nodiscard]] TensorPtr scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg);

/// General tensor contraction, connecting arbitrary pairs of (matching!) legs.
///
/// Parameters
/// ----------
/// tensor1, tensor2 : TensorCPtr
///     Tensors to contract.
/// legs1, legs2 : std::vector<LegRef>
///     Matching legs of ``tensor1`` and ``tensor2`` to contract.
/// relabel1, relabel2 : std::optional<std::map<std::string, std::string>>
///     Optional label maps. ``nullopt`` means no relabel.
///
/// Returns
/// -------
/// std::variant<TensorPtr, BlockBackend::Scalar>
///     Contracted tensor, or a scalar if no open legs remain.
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
