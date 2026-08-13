#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/tensor.h>

#include "../py_cyten_pybind11.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

namespace {

LegRef
py_as_leg_ref(py::object obj)
{
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    return obj.cast<int64>();
}

std::vector<LegRef>
py_as_leg_refs(py::object obj)
{
    std::vector<LegRef> out;
    if (py::isinstance<py::str>(obj) || !py::isinstance<py::iterable>(obj) ||
        py::isinstance<py::dict>(obj)) {
        out.push_back(py_as_leg_ref(obj));
        return out;
    }
    for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
        out.push_back(py_as_leg_ref(py::reinterpret_borrow<py::object>(item)));
    }
    return out;
}

std::optional<std::vector<LegRef>>
py_opt_leg_refs(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_as_leg_refs(obj);
}

int64
py_leg_idx(TensorCPtr const& tensor, py::object key)
{
    if (py::isinstance<py::str>(key)) {
        return tensor->get_leg_idcs(key.cast<std::string>())[0];
    }
    return tensor->get_leg_idcs(key.cast<int64>())[0];
}

std::optional<LevelsSpec>
py_as_levels(TensorCPtr const& tensor, py::object levels)
{
    if (levels.is_none()) {
        return std::nullopt;
    }
    LevelsSpec out;
    if (py::isinstance<py::dict>(levels)) {
        out.assign(static_cast<std::size_t>(tensor->num_legs), std::nullopt);
        py::dict d = py::reinterpret_borrow<py::dict>(levels);
        for (auto item : d) {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            py::object val = py::reinterpret_borrow<py::object>(item.second);
            int64 idx = py_leg_idx(tensor, key);
            if (val.is_none()) {
                out[static_cast<std::size_t>(idx)] = std::nullopt;
            } else {
                out[static_cast<std::size_t>(idx)] = val.cast<int64>();
            }
        }
        return out;
    }
    for (auto item : levels) {
        py::object o = py::reinterpret_borrow<py::object>(item);
        if (o.is_none()) {
            out.push_back(std::nullopt);
        } else {
            out.push_back(o.cast<int64>());
        }
    }
    return out;
}

std::optional<BendRight>
py_as_bend_right(TensorCPtr const& tensor, py::object bend_right)
{
    if (bend_right.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::bool_>(bend_right) ||
        py::isinstance(bend_right, py::module_::import("numpy").attr("bool_"))) {
        return bend_right.cast<bool>();
    }
    if (py::isinstance<py::dict>(bend_right)) {
        std::vector<std::optional<bool>> out(static_cast<std::size_t>(tensor->num_legs),
                                             std::nullopt);
        py::dict d = py::reinterpret_borrow<py::dict>(bend_right);
        for (auto item : d) {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            py::object val = py::reinterpret_borrow<py::object>(item.second);
            int64 idx = py_leg_idx(tensor, key);
            if (val.is_none()) {
                out[static_cast<std::size_t>(idx)] = std::nullopt;
            } else {
                out[static_cast<std::size_t>(idx)] = val.cast<bool>();
            }
        }
        return out;
    }
    if (py::isinstance<py::iterable>(bend_right) && !py::isinstance<py::str>(bend_right)) {
        std::vector<std::optional<bool>> out;
        for (auto item : bend_right) {
            py::object o = py::reinterpret_borrow<py::object>(item);
            if (o.is_none()) {
                out.push_back(std::nullopt);
            } else {
                out.push_back(o.cast<bool>());
            }
        }
        return out;
    }
    return bend_right.cast<bool>();
}

std::optional<PipeDualities>
py_as_pipe_dualities(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::bool_>(obj) ||
        py::isinstance(obj, py::module_::import("numpy").attr("bool_")) ||
        !py::isinstance<py::iterable>(obj) || py::isinstance<py::str>(obj)) {
        return obj.cast<bool>();
    }
    std::vector<bool> out;
    for (auto item : obj) {
        out.push_back(py::reinterpret_borrow<py::object>(item).cast<bool>());
    }
    return out;
}

std::optional<std::vector<Leg::Ptr>>
py_as_pipes(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<Leg::Ptr> out;
    for (auto item : obj) {
        py::object o = py::reinterpret_borrow<py::object>(item);
        if (o.is_none()) {
            out.push_back(nullptr);
        } else {
            out.push_back(o.cast<Leg::Ptr>());
        }
    }
    return out;
}

} // namespace

void
bind_tensors_ops_legs(py::module_& m)
{
    m.def(
      "bend_legs",
      [](TensorCPtr tensor, py::object num_codomain_legs, py::object num_domain_legs) {
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
      [](TensorCPtr tensor,
         py::args which_legs,
         py::object pipe_dualities,
         py::object pipes,
         py::object levels) {
          std::vector<std::vector<LegRef>> groups;
          groups.reserve(static_cast<std::size_t>(which_legs.size()));
          for (auto item : which_legs) {
              groups.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          auto dualities = py_as_pipe_dualities(pipe_dualities);
          auto pipes_opt = py_as_pipes(pipes);
          auto levels_opt = py_as_levels(tensor, levels);
          return combine_legs(std::move(tensor),
                              std::move(groups),
                              std::move(dualities),
                              std::move(pipes_opt),
                              std::move(levels_opt));
      },
      py::arg("tensor"),
      py::kw_only(),
      py::arg("pipe_dualities") = false,
      py::arg("pipes") = py::none(),
      py::arg("levels") = py::none(),
      R"pydoc(Combine (multiple) groups of legs, each to a :class:`LegPipe`.)pydoc");

    m.def(
      "combine_to_matrix",
      [](TensorCPtr tensor, py::object codomain, py::object domain, py::object levels) {
          return combine_to_matrix(tensor,
                                   py_opt_leg_refs(codomain),
                                   py_opt_leg_refs(domain),
                                   py_as_levels(tensor, levels));
      },
      py::arg("tensor"),
      py::arg("codomain") = py::none(),
      py::arg("domain") = py::none(),
      py::arg("levels") = py::none(),
      R"pydoc(Combine legs of a tensor into two combined LegPipes.)pydoc");

    m.def(
      "move_leg",
      [](TensorCPtr tensor,
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
          auto levels_opt = py_as_levels(tensor, levels);
          auto bend_opt = py_as_bend_right(tensor, bend_right);
          return move_leg(std::move(tensor),
                          py_as_leg_ref(which_leg),
                          cpos,
                          dpos,
                          std::move(levels_opt),
                          std::move(bend_opt));
      },
      py::arg("tensor"),
      py::arg("which_leg"),
      py::arg("codomain_pos") = py::none(),
      py::kw_only(),
      py::arg("domain_pos") = py::none(),
      py::arg("levels") = py::none(),
      py::arg("bend_right") = py::none(),
      R"pydoc(Move one leg of a tensor to a specified position.)pydoc");

    m.def(
      "permute_legs",
      [](TensorCPtr tensor,
         py::object codomain,
         py::object domain,
         py::object levels,
         py::object bend_right) {
          auto levels_opt = py_as_levels(tensor, levels);
          auto bend_opt = py_as_bend_right(tensor, bend_right);
          return permute_legs(std::move(tensor),
                              py_opt_leg_refs(codomain),
                              py_opt_leg_refs(domain),
                              std::move(levels_opt),
                              std::move(bend_opt));
      },
      py::arg("tensor"),
      py::arg("codomain") = py::none(),
      py::arg("domain") = py::none(),
      py::arg("levels") = py::none(),
      py::arg("bend_right") = py::none(),
      R"pydoc(Permute the legs of a tensor by braiding legs and bending lines.)pydoc");

    m.def(
      "split_legs",
      [](TensorCPtr tensor, py::object legs) {
          std::optional<std::vector<LegRef>> legs_opt;
          if (!legs.is_none()) {
              legs_opt = py_as_leg_refs(legs);
          }
          return split_legs(std::move(tensor), legs_opt);
      },
      py::arg("tensor"),
      py::arg("legs") = py::none(),
      R"pydoc(Split legs that were previously combined using :func:`combine_legs`.)pydoc");

    m.def(
      "squeeze_legs",
      [](TensorCPtr tensor, py::object legs) {
          std::optional<std::vector<LegRef>> legs_opt;
          if (!legs.is_none()) {
              legs_opt = py_as_leg_refs(legs);
          }
          return squeeze_legs(std::move(tensor), legs_opt);
      },
      py::arg("tensor"),
      py::arg("legs") = py::none(),
      R"pydoc(Remove trivial legs.)pydoc");
}

} // namespace cyten
