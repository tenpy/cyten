#include <cyten/tensors/ops_algebra.h>

#include "../py_cyten_pybind11.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

namespace {

std::optional<std::map<std::string, std::string>>
optional_relabel(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<std::map<std::string, std::string>>();
}

} // namespace

void
bind_tensors_ops_algebra(py::module_& m)
{
    m.def("almost_equal",
          &almost_equal,
          py::arg("tensor_1"),
          py::arg("tensor_2"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
          py::arg("allow_different_types") = false,
          R"pydoc(
Checks if two tensors are equal up to numerical tolerance.

We compare the blocks, i.e. the free parameters of the tensors.
The tensors count as almost equal if all block-entries, i.e. all their free parameters
individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
)pydoc");

    m.def(
      "apply_mask",
      &apply_mask,
      py::arg("tensor"),
      py::arg("mask"),
      py::arg("leg"),
      R"pydoc(Apply a projection Mask to one leg of a tensor, *projecting* it to a smaller leg.)pydoc");

    m.def(
      "enlarge_leg",
      &enlarge_leg,
      py::arg("tensor"),
      py::arg("mask"),
      py::arg("leg"),
      R"pydoc(Apply an inclusion Mask to one leg of a tensor *embedding* it into a larger leg.)pydoc");

    m.def("dagger",
          &dagger,
          py::arg("tensor"),
          R"pydoc(The hermitian conjugate tensor, a.k.a the dagger of a tensor.)pydoc");

    m.def(
      "compose",
      [](py::object tensor1, py::object tensor2, py::object relabel1, py::object relabel2) {
          return compose(std::move(tensor1),
                         std::move(tensor2),
                         optional_relabel(relabel1),
                         optional_relabel(relabel2));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      R"pydoc(Tensor contraction as map composition. Requires ``tensor1.domain == tensor2.codomain``.)pydoc");

    m.def("get_same_device",
          &get_same_device,
          py::kw_only(),
          py::arg("error_msg") = "Incompatible devices.",
          R"pydoc(If the given tensors have the same device, return it. Raise otherwise.)pydoc");

    m.def("inner",
          &inner,
          py::arg("A"),
          py::arg("B"),
          py::arg("do_dagger") = true,
          R"pydoc(The Frobenius inner product of two tensors.)pydoc");

    m.def("is_scalar", &is_scalar, py::arg("obj"), R"pydoc(If an object is a scalar.)pydoc");

    m.def("item",
          &item,
          py::arg("tensor"),
          R"pydoc(If the tensor is a scalar (with only trivial legs), convert to a Scalar.)pydoc");

    m.def("linear_combination",
          &linear_combination,
          py::arg("a"),
          py::arg("v"),
          py::arg("b"),
          py::arg("w"),
          R"pydoc(The linear combination ``a * v + b * w``)pydoc");

    m.def("norm", &norm, py::arg("tensor"), R"pydoc(The Frobenius norm of a Tensor.)pydoc");

    m.def("on_device",
          &on_device,
          py::arg("tensor"),
          py::arg("device"),
          py::arg("copy") = true,
          R"pydoc(An equivalent tensor (with the same entries) on another device.)pydoc");

    m.def(
      "outer",
      [](py::object tensor1, py::object tensor2, py::object relabel1, py::object relabel2) {
          return outer(std::move(tensor1),
                       std::move(tensor2),
                       optional_relabel(relabel1),
                       optional_relabel(relabel2));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      R"pydoc(The outer product, or tensor product.)pydoc");

    m.def(
      "partial_compose",
      [](py::object tensor1,
         py::object tensor2,
         py::object tensor1_first_leg,
         py::object relabel1,
         py::object relabel2) {
          return partial_compose(std::move(tensor1),
                                 std::move(tensor2),
                                 std::move(tensor1_first_leg),
                                 optional_relabel(relabel1),
                                 optional_relabel(relabel2));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("tensor1_first_leg"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      R"pydoc(Tensor contraction / composition involving only a part of the full (co)domain.)pydoc");

    m.def(
      "partial_trace",
      [](py::object tensor, py::args pairs, py::object levels) {
          std::vector<py::object> pair_vec;
          pair_vec.reserve(static_cast<std::size_t>(pairs.size()));
          for (auto item : pairs) {
              pair_vec.push_back(py::reinterpret_borrow<py::object>(item));
          }
          return partial_trace(std::move(tensor), std::move(pair_vec), std::move(levels));
      },
      py::arg("tensor"),
      py::kw_only(),
      py::arg("levels") = py::none(),
      R"pydoc(Perform a partial trace.)pydoc");

    m.def("pinv",
          &pinv,
          py::arg("tensor"),
          py::arg("cutoff") = 1e-15,
          R"pydoc(The Moore-Penrose pseudo-inverse of a tensor.)pydoc");

    m.def("scalar_multiply",
          &scalar_multiply,
          py::arg("a"),
          py::arg("v"),
          R"pydoc(The scalar multiplication ``a * v``)pydoc");

    m.def("scale_axis",
          &scale_axis,
          py::arg("tensor"),
          py::arg("diag"),
          py::arg("leg"),
          R"pydoc(Contract one `leg` of  `tensor` with a diagonal tensor.)pydoc");

    m.def(
      "tdot",
      [](py::object tensor1,
         py::object tensor2,
         py::object legs1,
         py::object legs2,
         py::object relabel1,
         py::object relabel2) {
          return tdot(std::move(tensor1),
                      std::move(tensor2),
                      std::move(legs1),
                      std::move(legs2),
                      optional_relabel(relabel1),
                      optional_relabel(relabel2));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("legs1"),
      py::arg("legs2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      R"pydoc(General tensor contraction, connecting arbitrary pairs of (matching!) legs.)pydoc");

    m.def("trace", &trace, py::arg("tensor"), R"pydoc(Perform the trace.)pydoc");

    m.def("transpose", &transpose, py::arg("tensor"), R"pydoc(The transpose of a tensor.)pydoc");
}

} // namespace cyten
