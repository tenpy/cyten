#pragma once

#include <cyten/cyten.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

/// Apply a mask to *both* legs of a diagonal tensor.
[[nodiscard]] py::object apply_mask_DiagonalTensor(py::object tensor, py::object mask);

/// The eigen-decomposition of a hermitian tensor.
[[nodiscard]] std::tuple<py::object, py::object> eigh(py::object tensor,
                                                      py::object new_labels,
                                                      bool new_leg_dual,
                                                      py::object sort = py::none());

/// The entropy of a probability distribution.
[[nodiscard]] py::object entropy(py::object p, py::object n = py::int_(1));

/// The LQ decomposition of a tensor.
[[nodiscard]] std::tuple<py::object, py::object> lq(py::object tensor,
                                                    py::object new_labels = py::none(),
                                                    bool new_leg_dual = false,
                                                    bool charge_leg_top = true);

/// The QR decomposition of a tensor.
[[nodiscard]] std::tuple<py::object, py::object> qr(py::object tensor,
                                                    py::object new_labels = py::none(),
                                                    bool new_leg_dual = false,
                                                    bool charge_leg_top = true);

/// The singular value decomposition (SVD) of a tensor.
[[nodiscard]] std::tuple<py::object, py::object, py::object> svd(
  py::object tensor,
  py::object new_labels = py::none(),
  bool new_leg_dual = false,
  bool charge_leg_top = true,
  py::object algorithm = py::none());

/// Truncate an existing SVD.
[[nodiscard]] std::tuple<py::object, py::object, py::object> svd_apply_mask(py::object U,
                                                                            py::object S,
                                                                            py::object Vh,
                                                                            py::object mask);

/// Given *normalized* singular values, determine which to keep.
[[nodiscard]] std::tuple<py::object, float64, float64> truncate_singular_values(
  py::object S,
  std::optional<int64> chi_max = std::nullopt,
  int64 chi_min = 1,
  float64 degeneracy_tol = 0.,
  float64 trunc_cut = 0.,
  float64 svd_min = 0.,
  bool minimize_error = true,
  py::object mask_labels = py::none());

/// Truncated version of :func:`svd`.
[[nodiscard]] std::tuple<py::object, py::object, py::object, float64, float64> truncated_svd(
  py::object tensor,
  py::object new_labels = py::none(),
  bool new_leg_dual = false,
  bool charge_leg_top = true,
  py::object algorithm = py::none(),
  std::optional<float64> normalize_to = std::nullopt,
  std::optional<int64> chi_max = std::nullopt,
  int64 chi_min = 1,
  float64 degeneracy_tol = 0.,
  float64 trunc_cut = 0.,
  float64 svd_min = 0.);

} // namespace cyten
