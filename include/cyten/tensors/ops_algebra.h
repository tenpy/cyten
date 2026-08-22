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
/// The mask must be a projection, i.e. its large leg is in its domain, at the top.
/// We apply the mask via map composition::
///
///     |                                │   │   ╭───╮   │
///     |      │   │   │                 │   │  ┏┷┓  │   │           │ ┏━┷━┓ │
///     |     ┏┷━━━┷━━━┷┓                │   │  ┃M┃  │   │           │ ┃M.T┃ │
///     |     ┃ tensor  ┃                │   │  ┗┯┛  │   │           │ ┗━┯━┛ │
///     |     ┗┯━━━┯━━━┯┛       OR       │   ╰───╯   │   │    ==    ┏┷━━━┷━━━┷┓
///     |      │   │  ┏┷┓               ┏┷━━━━━━━━━━━┷━━━┷┓         ┃ tensor  ┃
///     |      │   │  ┃M┃               ┃ tensor          ┃         ┗┯━━━┯━━━┯┛
///     |      │   │  ┗┯┛               ┗┯━━━┯━━━┯━━━┯━━━┯┛          │   │   │
///     |                                │   │   │   │   │
///
/// where ``M.T == transpose(M)``.
///
/// @param tensor The tensor to project.
/// @param mask Projection mask on `leg`.
/// @param leg Leg index or label to project.
[[nodiscard]] TensorPtr apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.
///
/// The mask must be an inclusion, i.e. its large leg is in its codomain, at the top.
/// We apply the mask via map composition::
///
///     |                                │   │   ╭───╮   │
///     |      │   │   │                 │   │  ┏┷┓  │   │           │ ┏━┷━┓ │
///     |     ┏┷━━━┷━━━┷┓                │   │  ┃M┃  │   │           │ ┃M.T┃ │
///     |     ┃ tensor  ┃                │   │  ┗┯┛  │   │           │ ┗━┯━┛ │
///     |     ┗┯━━━┯━━━┯┛       OR       │   ╰───╯   │   │    ==    ┏┷━━━┷━━━┷┓
///     |      │   │  ┏┷┓               ┏┷━━━━━━━━━━━┷━━━┷┓         ┃ tensor  ┃
///     |      │   │  ┃M┃               ┃ tensor          ┃         ┗┯━━━┯━━━┯┛
///     |      │   │  ┗┯┛               ┗┯━━━┯━━━┯━━━┯━━━┯┛          │   │   │
///     |                                │   │   │   │   │
///
/// where ``M.T == transpose(M)``.
///
/// @param tensor The tensor to enlarge.
/// @param mask Inclusion mask on `leg`.
/// @param leg Leg index or label to enlarge.
[[nodiscard]] TensorPtr enlarge_leg(TensorCPtr tensor, MaskCPtr mask, LegRef leg);

/// The hermitian conjugate tensor, a.k.a the dagger of a tensor.
/// For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
/// the hermitian conjugate matrix @f$ (M^\dagger)_{i,j} = \bar{M}_{j, i} @f$.
/// For a tensor ``A: W -> V`` the dagger is a map ``dagger(A): V -> W``.
/// Graphically::
///
///     |          e   d             a   b   c
///     |          │   │             │   │   │
///     |       ┏━━┷━━━┷━━┓         ┏┷━━━┷━━━┷┓
///     |       ┃    A    ┃         ┃dagger(A)┃
///     |       ┗┯━━━┯━━━┯┛         ┗━━┯━━━┯━━┛
///     |        │   │   │             │   │
///     |        a   b   c             e   d
///
/// Where ``a, b, c, d, e`` denote the legs in to (co-)domain.
///
/// @returns The hermitian conjugate tensor. Its legs and labels are::
///
///     dagger(A).codomain == A.domain
///     dagger(A).domain == A.codomain
///     dagger(A).legs == [leg.dual for leg in reversed(A.legs)]
///     dagger(A).labels == [_dual_leg_label(l) for l in reversed(A.labels)]
///
/// Note that the resulting `legs` only depend on the input `legs`, not
/// on their bipartition into domain and codomain.
/// For labels, we toggle a duality marker, i.e. if ``A.labels == ['a', 'b', 'c', 'd*', 'e*']``,
/// then ``dagger(A).labels == ['e', 'd', 'c*', 'b*','a*']``.
///
[[nodiscard]] TensorPtr dagger(TensorCPtr tensor);

/// Tensor contraction as map composition. Requires `tensor1.domain == tensor2.codomain`.
///  Graphically::
///
///  |        │   │   │   │
///  |       ┏┷━━━┷━━━┷━━━┷┓
///  |       ┃   tensor2   ┃
///  |       ┗━━━━┯━━━┯━━━━┛
///  |            │   │
///  |       ┏━━━━┷━━━┷━━━━┓
///  |       ┃   tensor1   ┃
///  |       ┗━━┯━━━┯━━━┯━━┛
///  |          │   │   │
///
/// If both tensors have no remaining open legs, returns a scalar.
///
/// @param tensor1,tensor2 Maps to compose: `tensor1` after `tensor2`.
/// @param relabel1,relabel2 Optional label maps applied before composition.
///     `nullopt` means no relabel.
/// @returns The composite map @f$ T_1 \circ T_2 @f$ from ``tensor2.domain`` to
/// ``tensor1.codomain``.

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
/// Graphically::
///
///    |          ╭───────────╮
///    |          │   ╭─────╮ │
///    |       ┏━━┷━━━┷━━┓  │ │
///    |       ┃    B    ┃  │ │
///    |       ┗┯━━━┯━━━┯┛  │ │
///    |       ┏┷━━━┷━━━┷┓  │ │
///    |       ┃dagger(A)┃  │ │
///    |       ┗━━┯━━━┯━━┛  │ │
///    |          │   ╰─────╯ │
///    |          ╰───────────╯
///
/// Assumes that the two tensors have the same (co-)domains.
/// The inner product is defined as @f$ \mathrm{Tr}[ A^\dagger \circ B] @f$.
/// It is thus equivalent to, but more efficient than ``trace(dot(A.hc, B))``.
///
/// @param A,B Tensors to take the inner product of.
/// @param do_dagger If ``True``, the standard inner product as above is computed.
///     If ``False``, we assume that the dagger has already been performed on one of the tensors.
///     Thus we require ``tensor_1.domain == tensor_2.codomain`` and vice versa and just perform
///     the contraction and trace.
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
///
/// The norm is given by @f$ \Vert A \Vert_\text{F} = \sqrt{\langle A \vert A \rangle_\text{F}}
/// @f$, where @f$ \langle {-} \vert {-} \rangle_\text{F} @f$ is the Frobenius inner product,
/// implemented in `inner`.
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
/// The outer product of two maps @f$ A : W_A \to V_A @f$ and @f$ B : W_B \to V_B @f$ is
/// a map @f$ A \otimes B : W_A \otimes W_B \to V_A \otimes V_B @f$.
///
///    |        │   │   │   │            │   │     │   │
///    |       ┏┷━━━┷━━━┷━━━┷┓          ┏┷━━━┷┓   ┏┷━━━┷┓
///    |       ┃ outer(A, B) ┃    ==    ┃  A  ┃   ┃  B  ┃
///    |       ┗━━┯━━━┯━━━┯━━┛          ┗┯━━━┯┛   ┗━━┯━━┛
///    |          │   │   │              │   │       │
///
///
/// @param tensor1,tensor2 Factors of the outer product.
/// @param relabel1,relabel2 Optional label maps. `nullopt` means no relabel.
//      The result has labels, as if the input tensors were relabelled accordingly before
//      contraction.
/// @returns The outer product @f$ A \otimes B @f$, with domain `[*A.domain, *B.domain]` and
/// codomain
///     `[*A.codomain, *B.codomain]`. Thus, the `Tensor.legs` are, *up to a permutation*,
///     the `Tensor.legs` of `A` plus the `Tensor.legs` of `B`.
[[nodiscard]] TensorPtr outer(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Tensor contraction / composition involving only a part of the full (co)domain.
///
/// Requires that all codomain (domain) legs of `tensor2` are consistent with the respective domain
/// (codomain) legs of `tensor1`; all legs to be contracted must be either in the codomain or in
/// the domain and `tensor1` must have at least one leg in the domain (codomain) that is not
/// contracted.
///
/// Graphically::
///
///     |        │   │   │   │
///     |       ┏┷━━━┷━━━┷━━━┷┓
///     |       ┃      A      ┃ == partial_compose(A, B, 2)
///     |       ┗┯━━━┯━━━┯━━━┯┛
///     |        │   │  ┏┷━━━┷┓
///     |        │   │  ┃  B  ┃
///     |        │   │  ┗┯━━━┯┛
///
/// Or::
///
///     |        │   │  ┏┷━━━┷┓
///     |        │   │  ┃  B  ┃
///     |        │   │  ┗┯━━━┯┛
///     |       ┏┷━━━┷━━━┷━━━┷┓
///     |       ┃      A      ┃ == partial_compose(A, B, 4)
///     |       ┗┯━━━┯━━━┯━━━┯┛
///     |        │   │   │   │
///
/// @param tensor1,tensor2 Tensors to partially compose.
/// @param tensor1_first_leg Which leg of `tensor1` is the first to be contracted with the first
/// leg of `tensor2`.
///     In particular, if `tensor1_first_leg < tensor1.num_codomain_legs`, part of the codomain
///     of `tensor1` is contracted with the full domain of `tensor2`, where
///     `tensor1.codomain[tensor1_first_leg] == tensor2.domain[0]`.
///     Otherwise (`tensor1_first_leg >= tensor1.num_codomain_legs`), part of the domain of
///     `tensor1` is contracted with the full codomain of `tensor2`, where
///     `tensor1.domain[tensor1.num_legs - 1 - tensor1_first_leg] == tensor2.codomain[-1]`.
/// @param relabel1,relabel2 Optional label maps. `nullopt` means no relabel.
/// @returns The partially composed tensor. The resulting legs correspond to the legs of `tensor1`
/// after
///     replacing the legs to be contracted by the open legs of `tensor2`.
[[nodiscard]] TensorPtr partial_compose(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  LegRef tensor1_first_leg,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform a partial trace over pairs of legs.
///
/// An arbitrary number of pairs can be traced over::
///
///     |    │       ╭───────╮
///     |    │   ╭───│───╮   │
///     |    7   6   5   4   │
///     |   ┏┷━━━┷━━━┷━━━┷┓  │
///     |   ┃      A      ┃  │    ==   partial_trace(A, (0, 2), (3, 5), (-2, 4))
///     |   ┗┯━━━┯━━━┯━━━┯┛  │
///     |    0   1   2   3   │
///     |    ╰───│───╯   ╰───╯
///
/// Note that despite its name, a "full" trace with a scalar result *can* be realized.
///
/// @param tensor Tensor to trace.
/// @param pairs A number of pairs, each describing two legs via index or via label.
///     Each pair is connected, realizing a partial trace.
///     By definition, we create loops between legs on opposite sides to the right side of the
///     tensor (this is not necessarily equivalent to a left closing, if there are braids).
///     Must be compatible ``tensor.get_leg(pair[0]) == tensor.get_leg(pair[1]).dual``.
/// @param levels The connectivity of the partial trace may induce braids.
///     For symmetries with non-symmetric braiding, these levels are used to determine the
///     chirality of those braids, like in :func:`permute_legs`.
/// @returns If all legs are traced, a python scalar.
///     If legs are left open, a tensor with the same type as `tensor`.
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
/// Leg order, labels and legs of `tensor` are not changed.
/// The diagonal tensors leg ``diag.leg`` must be the same or the dual of the leg on the tensor,
/// if mismatched, the `diag` is automatically transposed, as needed.
///
/// Graphically::
///
///     |        │   │   │            │   │  ┏┷┓
///     |       ┏┷━━━┷━━━┷┓           │   │  ┃D┃
///     |       ┃ tensor  ┃           │   │  ┗┯┛
///     |       ┗┯━━━┯━━━┯┛    OR    ┏┷━━━┷━━━┷┓
///     |        │  ┏┷┓  │           ┃ tensor  ┃
///     |        │  ┃D┃  │           ┗┯━━━┯━━━┯┛
///     |        │  ┗┯┛  │            │   │   │
///
/// Or transpose as needed:
///
///     |        │   │   │   │   │
///     |       ┏┷━━━┷━━━┷━━━┷━━━┷┓            │   │   │
///     |       ┃ tensor          ┃           ┏┷━━━┷━━━┷┓
///     |       ┗┯━━━┯━━━━━━━━━━━┯┛           ┃ tensor  ┃
///     |        │   │   ╭───╮   │      ==    ┗┯━━━┯━━━┯┛
///     |        │   │  ┏┷┓  │   │             │ ┏━┷━┓ │
///     |        │   │  ┃D┃  │   │             │ ┃D.T┃ │
///     |        │   │  ┗┯┛  │   │             │ ┗━┯━┛ │
///     |        │   ╰───╯   │   │
///
/// where ``D.T == transpose(D)``.
///
/// @param tensor Tensor to scale along one axis.
/// @param diag Diagonal tensor contracted onto `leg`.
/// @param leg Leg index or label to contract.
[[nodiscard]] TensorPtr scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg);

/// General tensor contraction, connecting arbitrary pairs of (matching!) legs.
///
/// For example::
///
///     |    ╭───╮   ╭───│───│──╮
///     |    │   4   3   2   │  │
///     |    │  ┏┷━━━┷━━━┷┓  │  │
///     |    │  ┃    B    ┃  │  │
///     |    │  ┗━━┯━━━┯━━┛  │  │
///     |    │     0   1     │  │
///     |    │     │   ╰─────╯  │    ==    tdot(A, B, [1, 4, 5], [3, 0, 4])
///     |    ╰───╮ ╰─╮   ╭───╮  │
///     |        5   4   3   │  │
///     |       ┏┷━━━┷━━━┷┓  │  │
///     |       ┃    A    ┃  │  │
///     |       ┗┯━━━┯━━━┯┛  │  │
///     |        0   1   2   │  │
///     |        │   ╰───│───│──╯
///
/// @param tensor1,tensor2 Tensors to contract.
/// @param legs1,legs2 Which legs to contract: `legs1[n]` on `tensor1` is contracted with
/// `legs2[n]` on
///     `tensor2`.
/// @param relabel1,relabel2 A mapping of labels for each of the tensors.
///     The result has labels as if the input tensors were relabelled accordingly before
///     contraction.
/// @returns A tensor given by the contraction.
///    Its domain is formed by the uncontracted legs of `tensor2`, in *inverse* order and with
///    *opposite* duality compared to ``tensor2.legs``, i.e. like they were all in
///    ``tensor2.domain``. Its codomain, conversely, is given by the uncontracted legs of
///    `tensor1`, in the same order and with the same duality as in ``tensor1.legs``, i.e. like
///    they were all in ``tensor1.codomain``. Therefore, the ``result.legs`` are the uncontracted
///    from ``tensor1.legs``, followed by the uncontracted ``tensor2.legs``.
[[nodiscard]] std::variant<TensorPtr, BlockBackend::Scalar> tdot(
  TensorCPtr tensor1,
  TensorCPtr tensor2,
  std::vector<LegRef> legs1,
  std::vector<LegRef> legs2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform the full trace.
///
/// Requires that ``tensor.domain == tensor.codomain`` and perform the full trace::
///
///     |    ╭───────────────╮
///     |    │   ╭─────────╮ │
///     |    │   │   ╭───╮ │ │
///     |   ┏┷━━━┷━━━┷┓  │ │ │
///     |   ┃    A    ┃  │ │ │    ==    trace(A)
///     |   ┗┯━━━┯━━━┯┛  │ │ │
///     |    │   │   ╰───╯ │ │
///     |    │   ╰─────────╯ │
///     |    ╰───────────────╯
/// @param tensor Tensor to trace.
/// @returns A single scalar, the trace.
///
[[nodiscard]] BlockBackend::Scalar trace(TensorCPtr tensor);

/// The transpose of a tensor.
///
/// For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
/// the transpose matrix :math:`(M^\text{T})_{i,j} = M_{j, i}` .
/// For a map :math:`f: V \to W`, the transpose is a map :math:`f: W^* \to V^*`::
///
///     |          │   │   │             ╭───────────╮
///     |          │   │   │             │ ╭─────╮   │     │ │ │
///     |       ┏━━┷━━━┷━━━┷━━┓          │ │  ┏━━┷━━━┷━━┓  │ │ │
///     |       ┃transpose(A) ┃    ==    │ │  ┃    A    ┃  │ │ │
///     |       ┗━━━━┯━━━┯━━━━┛          │ │  ┗┯━━━┯━━━┯┛  │ │ │
///     |            │   │               │ │   │   │   ╰───╯ │ │
///     |            │   │               │ │   │   ╰─────────╯ │
///     |            │   │               │ │   ╰───────────────╯
///
/// @returns The transposed tensor. Its legs and labels fulfill e.g.::
///
///        transpose(A).codomain == A.domain.dual == [W2.dual, W1.dual]  # if A.domain == [W1, W2]
///        transpose(A).domain == A.codomain.dual == [V2.dual, V1.dual]  # if A.codomain == [V1,
///        V2] transpose(A).legs == [W2.dual, W1.dual, V1, V2]  # compared to A.legs == [V1, V2,
///        W2.dual, W1.dual] transpose(A).labels == [*reversed(A.domain_labels),
///        *A.codomain_labels]
///
///    Note that the resulting `Tensor.legs` depend not only on the input `Tensor.legs`,
///    but also on how they are partitioned into domain and codomain.
///    We use the "same" labels, up to the permutation.
[[nodiscard]] TensorPtr transpose(TensorCPtr tensor);

} // namespace cyten
