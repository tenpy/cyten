#pragma once

#include <cyten/cyten.h>

#include <optional>
#include <vector>

namespace cyten {

/// Move legs between codomain and domain without changing the order of ``tensor.legs``.
[[nodiscard]] py::object bend_legs(py::object tensor,
                                   std::optional<int64> num_codomain_legs = std::nullopt,
                                   std::optional<int64> num_domain_legs = std::nullopt);

/// Check if two tensors have the same legs.
void check_same_legs(py::object t1, py::object t2);

/// Combine (multiple) groups of legs, each to a :class:`LegPipe`.
[[nodiscard]] py::object combine_legs(py::object tensor,
                                      std::vector<py::object> which_legs,
                                      py::object pipe_dualities = py::none(),
                                      py::object pipes = py::none(),
                                      py::object levels = py::none());

/// Combine legs of a tensor into two combined LegPipes (matrix form).
[[nodiscard]] py::object combine_to_matrix(py::object tensor,
                                           py::object codomain = py::none(),
                                           py::object domain = py::none(),
                                           py::object levels = py::none());

/// Move one leg of a tensor to a specified position.
[[nodiscard]] py::object move_leg(py::object tensor,
                                  py::object which_leg,
                                  std::optional<int64> codomain_pos = std::nullopt,
                                  std::optional<int64> domain_pos = std::nullopt,
                                  py::object levels = py::none(),
                                  py::object bend_right = py::none());

/// Permute the legs of a tensor by braiding legs and bending lines.
[[nodiscard]] py::object permute_legs(py::object tensor,
                                      py::object codomain = py::none(),
                                      py::object domain = py::none(),
                                      py::object levels = py::none(),
                                      py::object bend_right = py::none());

/// Split legs that were previously combined using :func:`combine_legs`.
[[nodiscard]] py::object split_legs(py::object tensor, py::object legs = py::none());

/// Remove trivial legs.
[[nodiscard]] py::object squeeze_legs(py::object tensor, py::object legs = py::none());

} // namespace cyten
