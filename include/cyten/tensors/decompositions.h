#pragma once

#include <cyten/cyten.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/ops_algebra.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

/// Apply a mask to *both* legs of a diagonal tensor.
[[nodiscard]] DiagonalTensorPtr apply_mask_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask);

/// The eigen-decomposition of a hermitian tensor.
[[nodiscard]] std::tuple<DiagonalTensorPtr, TensorPtr> eigh(
  TensorCPtr tensor,
  LegLabels new_labels,
  bool new_leg_dual,
  std::optional<std::string> sort = std::nullopt);

/// The entropy of a probability distribution (tensor path).
[[nodiscard]] BlockBackend::Scalar entropy(DiagonalTensorCPtr p, float64 n = 1);

/// The LQ decomposition of a tensor.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> lq(TensorCPtr tensor,
                                                  std::optional<LegLabels> new_labels = std::nullopt,
                                                  bool new_leg_dual = false,
                                                  bool charge_leg_top = true);

/// The QR decomposition of a tensor.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> qr(TensorCPtr tensor,
                                                  std::optional<LegLabels> new_labels = std::nullopt,
                                                  bool new_leg_dual = false,
                                                  bool charge_leg_top = true);

/// The singular value decomposition (SVD) of a tensor.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr> svd(
  TensorCPtr tensor,
  std::optional<LegLabels> new_labels = std::nullopt,
  bool new_leg_dual = false,
  bool charge_leg_top = true,
  std::optional<std::string> algorithm = std::nullopt);

/// Truncate an existing SVD.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr> svd_apply_mask(TensorCPtr U,
                                                                                 DiagonalTensorCPtr S,
                                                                                 TensorCPtr Vh,
                                                                                 MaskCPtr mask);

/// Given *normalized* singular values, determine which to keep.
[[nodiscard]] std::tuple<MaskPtr, float64, float64> truncate_singular_values(
  DiagonalTensorCPtr S,
  std::optional<int64> chi_max = std::nullopt,
  int64 chi_min = 1,
  float64 degeneracy_tol = 0.,
  float64 trunc_cut = 0.,
  float64 svd_min = 0.,
  bool minimize_error = true,
  std::optional<LegLabels> mask_labels = std::nullopt);

/// Truncated version of :func:`svd`.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr, float64, float64> truncated_svd(
  TensorCPtr tensor,
  std::optional<LegLabels> new_labels = std::nullopt,
  bool new_leg_dual = false,
  bool charge_leg_top = true,
  std::optional<std::string> algorithm = std::nullopt,
  std::optional<float64> normalize_to = std::nullopt,
  std::optional<int64> chi_max = std::nullopt,
  int64 chi_min = 1,
  float64 degeneracy_tol = 0.,
  float64 trunc_cut = 0.,
  float64 svd_min = 0.);

} // namespace cyten
