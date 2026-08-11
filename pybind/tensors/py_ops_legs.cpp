#include <cyten/tensors/ops_legs.h>

#include "../py_cyten_pybind11.h"

#include <optional>
#include <vector>

namespace cyten {

void
bind_tensors_ops_legs(py::module_& m)
{
    m.def(
      "bend_legs",
      [](py::object tensor, py::object num_codomain_legs, py::object num_domain_legs) {
          std::optional<int64> n_cod;
          std::optional<int64> n_dom;
          if (!num_codomain_legs.is_none()) {
              n_cod = num_codomain_legs.cast<int64>();
          }
          if (!num_domain_legs.is_none()) {
              n_dom = num_domain_legs.cast<int64>();
          }
          return bend_legs(std::move(tensor), n_cod, n_dom);
      },
      py::arg("tensor"),
      py::arg("num_codomain_legs") = py::none(),
      py::arg("num_domain_legs") = py::none(),
      R"pydoc(Move legs between codomain and domain without changing the order of ``tensor.legs``.)pydoc");

    m.def("check_same_legs",
          &check_same_legs,
          py::arg("t1"),
          py::arg("t2"),
          R"pydoc(Check if two tensors have the same legs.)pydoc");

    m.def(
      "combine_legs",
      [](py::object tensor,
         py::args which_legs,
         py::object pipe_dualities,
         py::object pipes,
         py::object levels) {
          std::vector<py::object> groups;
          groups.reserve(static_cast<std::size_t>(which_legs.size()));
          for (auto item : which_legs) {
              groups.push_back(py::reinterpret_borrow<py::object>(item));
          }
          return combine_legs(std::move(tensor),
                              std::move(groups),
                              std::move(pipe_dualities),
                              std::move(pipes),
                              std::move(levels));
      },
      py::arg("tensor"),
      py::kw_only(),
      py::arg("pipe_dualities") = false,
      py::arg("pipes") = py::none(),
      py::arg("levels") = py::none(),
      R"pydoc(Combine (multiple) groups of legs, each to a :class:`LegPipe`.)pydoc");

    m.def("combine_to_matrix",
          &combine_to_matrix,
          py::arg("tensor"),
          py::arg("codomain") = py::none(),
          py::arg("domain") = py::none(),
          py::arg("levels") = py::none(),
          R"pydoc(Combine legs of a tensor into two combined LegPipes.)pydoc");

    m.def(
      "move_leg",
      [](py::object tensor,
         py::object which_leg,
         py::object codomain_pos,
         py::object domain_pos,
         py::object levels,
         py::object bend_right) {
          std::optional<int64> cpos;
          std::optional<int64> dpos;
          if (!codomain_pos.is_none()) {
              cpos = codomain_pos.cast<int64>();
          }
          if (!domain_pos.is_none()) {
              dpos = domain_pos.cast<int64>();
          }
          return move_leg(std::move(tensor),
                          std::move(which_leg),
                          cpos,
                          dpos,
                          std::move(levels),
                          std::move(bend_right));
      },
      py::arg("tensor"),
      py::arg("which_leg"),
      py::arg("codomain_pos") = py::none(),
      py::kw_only(),
      py::arg("domain_pos") = py::none(),
      py::arg("levels") = py::none(),
      py::arg("bend_right") = py::none(),
      R"pydoc(Move one leg of a tensor to a specified position.)pydoc");

    m.def("permute_legs",
          &permute_legs,
          py::arg("tensor"),
          py::arg("codomain") = py::none(),
          py::arg("domain") = py::none(),
          py::arg("levels") = py::none(),
          py::arg("bend_right") = py::none(),
          R"pydoc(Permute the legs of a tensor by braiding legs and bending lines.)pydoc");

    m.def("split_legs",
          &split_legs,
          py::arg("tensor"),
          py::arg("legs") = py::none(),
          R"pydoc(Split legs that were previously combined using :func:`combine_legs`.)pydoc");

    m.def("squeeze_legs",
          &squeeze_legs,
          py::arg("tensor"),
          py::arg("legs") = py::none(),
          R"pydoc(Remove trivial legs.)pydoc");
}

} // namespace cyten
