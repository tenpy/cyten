#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/labels.h>

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// The identity tensor on a given leg.
///
/// Returns a :class:`DiagonalTensor` if ``diagonal`` is true, otherwise a
/// :class:`SymmetricTensor`.
[[nodiscard]] py::object eye(py::object leg,
                             TensorBackend::Ptr backend = nullptr,
                             py::object labels = py::none(),
                             Dtype dtype = Dtype::Float64,
                             std::optional<std::string> device = std::nullopt,
                             bool diagonal = true);

/// Convert object to tensor if possible.
[[nodiscard]] py::object tensor(py::object obj,
                                py::object codomain,
                                py::object domain = py::none(),
                                TensorBackend::Ptr backend = nullptr,
                                py::object labels = py::none(),
                                std::optional<Dtype> dtype = std::nullopt,
                                std::optional<std::string> device = std::nullopt,
                                bool understood_braiding = false);

/// Add a trivial leg to a tensor.
///
/// A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.
///
/// Parameters
/// ----------
/// tens: Tensor
///     The tensor to add a leg to. Since :class:`DiagonalTensor` and :class:`Mask` do not
///     support adding legs, they will be converted to :class:`SymmetricTensor` first.
/// legs_pos, codomain_pos, domain_pos: int
///     The position of the new leg can be specified in three mutually exclusive ways.
///     If the positional argument `leg_pos` is used, ``result.legs[leg_pos]`` will be the trivial
///     leg. In most cases that unambiguously assigns it to either the domain or the codomain.
///     If ambiguous (``if legs_pos == num_codomain_legs``), it is added to the codomain.
///     Alternatively, it can be added to the codomain at ``codomain[codomain_pos]``
///     or to the domain at ``domain_pos``.
///     Note the implications for the ``is_dual`` argument!
///     Per default, we use ``0``, i.e. add at ``legs[0]`` / ``codomain[0]``.
/// label: str
///     The label for the new leg.
/// is_dual: bool
///     If we add a dual (bra-like) or ket-like leg.
///     Note that if `leg_pos` is given, we have ``result.legs[leg_pos].is_dual == is_dual``,
///     but if `domain_pos` is given, we have ``result.domain[domain_pos].is_dual == is_dual``,
///     which are mutually opposite.
[[nodiscard]] py::object add_trivial_leg(py::object tens,
                                         std::optional<int64> legs_pos = std::nullopt,
                                         std::optional<int64> codomain_pos = std::nullopt,
                                         std::optional<int64> domain_pos = std::nullopt,
                                         LegLabel label = std::nullopt,
                                         bool is_dual = false);

/// Return a zero tensor with same type, dtype, legs, backend and labels.
[[nodiscard]] py::object zero_like(py::object tensor);

/// Stack a grid of tensors along existing legs.
///
/// The tensors are stacked along the first leg in their codomain and the final leg in their
/// domain. The resulting legs are :math:`result.codomain[0] = V = \bigoplus_m V_m` and
/// :math:`result.domain[-1] = W = \bigoplus_n W_n`, where :math:`V_m` is the first codomain leg
/// of all tensors in the ``m``-th row ``grid[m]``, and :math:`W_n` is the last domain leg of all
/// tensors in the ``n``-th column, i.e. for the tensors ``[row[n] for row in grid]``.
[[nodiscard]] py::object tensor_from_grid(py::object grid,
                                          py::object labels = py::none(),
                                          std::optional<Dtype> dtype = std::nullopt);

} // namespace cyten
