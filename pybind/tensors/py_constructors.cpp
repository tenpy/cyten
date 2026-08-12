#include <cyten/tensors/constructors.h>

#include "../py_cyten_pybind11.h"

#include <optional>
#include <string>

namespace cyten {

void
bind_tensors_constructors(py::module_& m)
{
    m.def(
      "eye",
      [](py::object leg,
         TensorBackend::Ptr backend,
         py::object labels,
         Dtype dtype,
         py::object device,
         bool diagonal) {
          std::optional<std::string> device_opt;
          if (!device.is_none()) {
              device_opt = device.cast<std::string>();
          }
          return eye(
            std::move(leg), std::move(backend), std::move(labels), dtype, device_opt, diagonal);
      },
      py::arg("leg"),
      py::arg("backend") = py::none(),
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Float64,
      py::arg("device") = py::none(),
      py::arg("diagonal") = true,
      R"pydoc(The identity tensor on a given leg.)pydoc");

    m.def(
      "tensor",
      [](py::object obj,
         py::object codomain,
         py::object domain,
         TensorBackend::Ptr backend,
         py::object labels,
         py::object dtype,
         py::object device,
         bool understood_braiding) {
          std::optional<Dtype> dtype_opt;
          if (!dtype.is_none()) {
              dtype_opt = dtype.cast<Dtype>();
          }
          std::optional<std::string> device_opt;
          if (!device.is_none()) {
              device_opt = device.cast<std::string>();
          }
          return tensor(std::move(obj),
                        std::move(codomain),
                        std::move(domain),
                        std::move(backend),
                        std::move(labels),
                        dtype_opt,
                        device_opt,
                        understood_braiding);
      },
      py::arg("obj"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = py::none(),
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("understood_braiding") = false,
      R"pydoc(Convert object to tensor if possible.)pydoc");

    m.def(
      "add_trivial_leg",
      [](py::object tens,
         py::object legs_pos,
         py::object codomain_pos,
         py::object domain_pos,
         py::object label,
         bool is_dual) {
          std::optional<int64> legs_pos_opt;
          std::optional<int64> codomain_pos_opt;
          std::optional<int64> domain_pos_opt;
          LegLabel label_opt = std::nullopt;
          if (!legs_pos.is_none()) {
              legs_pos_opt = legs_pos.cast<int64>();
          }
          if (!codomain_pos.is_none()) {
              codomain_pos_opt = codomain_pos.cast<int64>();
          }
          if (!domain_pos.is_none()) {
              domain_pos_opt = domain_pos.cast<int64>();
          }
          if (!label.is_none()) {
              label_opt = label.cast<std::string>();
          }
          return add_trivial_leg(std::move(tens),
                                 legs_pos_opt,
                                 codomain_pos_opt,
                                 domain_pos_opt,
                                 std::move(label_opt),
                                 is_dual);
      },
      py::arg("tens"),
      py::arg("legs_pos") = py::none(),
      py::kw_only(),
      py::arg("codomain_pos") = py::none(),
      py::arg("domain_pos") = py::none(),
      py::arg("label") = py::none(),
      py::arg("is_dual") = false,
      R"pydoc(
Add a trivial leg to a tensor.

A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.

Parameters
----------
tens: Tensor
    The tensor to add a leg to. Since :class:`DiagonalTensor` and :class:`Mask` do not
    support adding legs, they will be converted to :class:`SymmetricTensor` first.
legs_pos, codomain_pos, domain_pos: int
    The position of the new leg can be specified in three mutually exclusive ways.
    If the positional argument `leg_pos` is used, ``result.legs[leg_pos]`` will be the trivial
    leg. In most cases that unambiguously assigns it to either the domain or the codomain.
    If ambiguous (``if legs_pos == num_codomain_legs``), it is added to the codomain.
    Alternatively, it can be added to the codomain at ``codomain[codomain_pos]``
    or to the domain at ``domain_pos``.
    Note the implications for the ``is_dual`` argument!
    Per default, we use ``0``, i.e. add at ``legs[0]`` / ``codomain[0]``.
label: str
    The label for the new leg.
is_dual: bool
    If we add a dual (bra-like) or ket-like leg.
    Note that if `leg_pos` is given, we have ``result.legs[leg_pos].is_dual == is_dual``,
    but if `domain_pos` is given, we have ``result.domain[domain_pos].is_dual == is_dual``,
    which are mutually opposite.
)pydoc");

    m.def("zero_like",
          &zero_like,
          py::arg("tensor"),
          R"pydoc(Return a zero tensor with same type, dtype, legs, backend and labels.)pydoc");

    m.def(
      "tensor_from_grid",
      [](py::object grid, py::object labels, py::object dtype) {
          std::optional<Dtype> dtype_opt;
          if (!dtype.is_none()) {
              dtype_opt = dtype.cast<Dtype>();
          }
          return tensor_from_grid(std::move(grid), std::move(labels), dtype_opt);
      },
      py::arg("grid"),
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      R"pydoc(
Stack a grid of tensors along existing legs.

The tensors are stacked along the first leg in their codomain and the final leg in their
domain. The resulting legs are :math:`result.codomain[0] = V = \bigoplus_m V_m` and
:math:`result.domain[-1] = W = \bigoplus_n W_n`, where :math:`V_m` is the first codomain leg
of all tensors in the ``m``-th row ``grid[m]``, and :math:`W_n` is the last domain leg of all
tensors in the ``n``-th column, i.e. for the tensors ``[row[n] for row in grid]``.

Graphically::

    |                                                      W
    |                                              │   │ ┏━┷━┓
    |                                              │   │ ┃p_n┃
    |                  W                           │   │ ┗━┯━┛
    |          │   │   │                           │   │   │ W_n
    |       ┏━━┷━━━┷━━━┷━━┓                     ┏━━┷━━━┷━━━┷━━┓
    |       ┃     res     ┃    ==   sum_{m,n}   ┃  grid[m][n] ┃
    |       ┗┯━━━┯━━━┯━━━┯┛                     ┗┯━━━┯━━━┯━━━┯┛
    |        │   │   │   │                   V_m │   │   │   │
    |        V                                 ┏━┷━┓ │   │   │
    |                                          ┃i_m┃ │   │   │
    |                                          ┗━┯━┛ │   │   │
    |

where :math:`p_n : W = \bigoplus_{n'} W_{n'} \to W_n` is the projection map of the direct sum
and :math:`i_m : V_m \to \bigoplus_{m'} V_{m'}` the inclusion.

Parameters
----------
grid: list[list[SymmetricTensor | None]]
    Contains the tensors from which a single tensor is constructed by stacking. `None` entries
    are interpreted as tensors with all blocks equal to zero. All legs except the ones along
    which the stacking happens must be identical across all tensors in the grid. For
    consistency, tensors within the same row must have identical left spaces (first leg in the
    codomain), and tensors within the same column must have identical right spaces (final leg
    in the domain).
labels
    Leg labels of the resulting tensor.
dtype: Dtype | None
    The dtype of the tensor. Uses the common dtype across all tensors in the grid if `None`.
)pydoc");
}

} // namespace cyten
