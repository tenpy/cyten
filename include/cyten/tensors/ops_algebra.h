#pragma once

#include <cyten/cyten.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Checks if two tensors are equal up to numerical tolerance.
[[nodiscard]] bool almost_equal(py::object tensor_1,
                                py::object tensor_2,
                                float64 rtol = 1e-5,
                                float64 atol = 1e-8,
                                bool allow_different_types = false);

/// Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.
[[nodiscard]] py::object apply_mask(py::object tensor, py::object mask, py::object leg);

/// Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.
[[nodiscard]] py::object enlarge_leg(py::object tensor, py::object mask, py::object leg);

/// The hermitian conjugate tensor, a.k.a the dagger of a tensor.
[[nodiscard]] py::object dagger(py::object tensor);

/// Tensor contraction as map composition. Requires ``tensor1.domain == tensor2.codomain``.
[[nodiscard]] py::object compose(
  py::object tensor1,
  py::object tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// If the given tensors have the same device, return it. Raise otherwise.
[[nodiscard]] std::string get_same_device(py::args tensors,
                                         std::string const& error_msg = "Incompatible devices.");

/// The Frobenius inner product of two tensors.
[[nodiscard]] py::object inner(py::object A, py::object B, bool do_dagger = true);

/// If an object is a scalar (number or single-sector tensor).
[[nodiscard]] bool is_scalar(py::object obj);

/// If the tensor is a scalar (with only trivial legs), convert to a Scalar.
[[nodiscard]] py::object item(py::object tensor);

/// The linear combination ``a * v + b * w``.
[[nodiscard]] py::object linear_combination(py::object a,
                                            py::object v,
                                            py::object b,
                                            py::object w);

/// The Frobenius norm of a Tensor.
[[nodiscard]] py::object norm(py::object tensor);

/// An equivalent tensor (with the same entries) on another device.
[[nodiscard]] py::object on_device(py::object tensor, std::string device, bool copy = true);

/// The outer product, or tensor product.
[[nodiscard]] py::object outer(
  py::object tensor1,
  py::object tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Tensor contraction / composition involving only a part of the full (co)domain.
[[nodiscard]] py::object partial_compose(
  py::object tensor1,
  py::object tensor2,
  py::object tensor1_first_leg,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform a partial trace over pairs of legs.
[[nodiscard]] py::object partial_trace(py::object tensor,
                                       std::vector<py::object> pairs,
                                       py::object levels = py::none());

/// The Moore-Penrose pseudo-inverse of a tensor.
[[nodiscard]] py::object pinv(py::object tensor, float64 cutoff = 1e-15);

/// The scalar multiplication ``a * v``.
[[nodiscard]] py::object scalar_multiply(py::object a, py::object v);

/// Contract one `leg` of `tensor` with a diagonal tensor.
[[nodiscard]] py::object scale_axis(py::object tensor, py::object diag, py::object leg);

/// General tensor contraction, connecting arbitrary pairs of (matching!) legs.
[[nodiscard]] py::object tdot(
  py::object tensor1,
  py::object tensor2,
  py::object legs1,
  py::object legs2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Perform the full trace. Requires ``tensor.domain == tensor.codomain``.
[[nodiscard]] py::object trace(py::object tensor);

/// The transpose of a tensor.
[[nodiscard]] py::object transpose(py::object tensor);

} // namespace cyten
