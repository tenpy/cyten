#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/direct_sum.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tensors/vector_like.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include "docstrings/tensors/ops_algebra.h"

#include <format>
#include <map>
#include <optional>
#include <string>
#include <variant>
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

py::object
py_from_tensor_or_scalar(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, v);
}

/// Downcast a VectorLike pointer so Tensor arithmetic still returns a Tensor.
py::object
py_cast_vector_like(VectorLikePtr p)
{
    if (!p) {
        return py::none();
    }
    if (auto t = std::dynamic_pointer_cast<Tensor>(p)) {
        return py::cast(std::move(t));
    }
    if (auto ds = std::dynamic_pointer_cast<DirectSum>(p)) {
        return py::cast(std::move(ds));
    }
    return py::cast(std::move(p));
}

BlockBackend::Scalar
py_as_scalar(py::object obj, TensorCPtr hint)
{
    try {
        return obj.cast<BlockBackend::Scalar>();
    } catch (py::cast_error const&) {
    }
    return hint->backend->block_backend->as_scalar(obj, hint->dtype);
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
          DOC(cyten, almost_equal));

    m.def(
      "apply_mask",
      [](TensorCPtr tensor, MaskCPtr mask, py::object leg) {
          return apply_mask(std::move(tensor), std::move(mask), py_as_leg_ref(leg));
      },
      py::arg("tensor"),
      py::arg("mask"),
      py::arg("leg"),
      DOC(cyten, apply_mask));

    m.def(
      "enlarge_leg",
      [](TensorCPtr tensor, MaskCPtr mask, py::object leg) {
          return enlarge_leg(std::move(tensor), std::move(mask), py_as_leg_ref(leg));
      },
      py::arg("tensor"),
      py::arg("mask"),
      py::arg("leg"),
      DOC(cyten, enlarge_leg));

    m.def("dagger", &dagger, py::arg("tensor"), DOC(cyten, dagger));

    m.def(
      "compose",
      [](TensorCPtr tensor1, TensorCPtr tensor2, py::object relabel1, py::object relabel2) {
          return py_from_tensor_or_scalar(compose(std::move(tensor1),
                                                  std::move(tensor2),
                                                  optional_relabel(relabel1),
                                                  optional_relabel(relabel2)));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      doc_plus(DOC(cyten, compose),
               R"pydoc(
In Python, ``relabel1`` / ``relabel2`` are ``dict | None`` (``None`` = no relabel).
)pydoc"));

    m.def(
      "get_same_device",
      [](py::args tensors, std::string error_msg) {
          std::vector<TensorCPtr> vec;
          vec.reserve(static_cast<std::size_t>(tensors.size()));
          for (auto item : tensors) {
              vec.push_back(item.cast<TensorCPtr>());
          }
          return get_same_device(vec, std::move(error_msg));
      },
      py::kw_only(),
      py::arg("error_msg") = "Incompatible devices.",
      DOC(cyten, get_same_device));

    // Bind the VectorLike overload (DOC(..., 2)); Tensor overload is DOC(cyten, inner).
    m.def("inner",
          static_cast<BlockBackend::Scalar (*)(VectorLikeCPtr, VectorLikeCPtr, bool)>(&inner),
          py::arg("A"),
          py::arg("B"),
          py::arg("do_dagger") = true,
          DOC(cyten, inner, 2));

    m.def(
      "is_scalar",
      [](py::object obj) {
          if (py::hasattr(obj, "domain") && py::hasattr(obj, "codomain")) {
              if (obj.attr("domain").attr("num_sectors").cast<int64>() != 1) {
                  return false;
              }
              if (obj.attr("codomain").attr("num_sectors").cast<int64>() != 1) {
                  return false;
              }
              auto np = py::module_::import("numpy");
              if (!np.attr("array_equal")(obj.attr("domain").attr("sector_decomposition"),
                                          obj.attr("codomain").attr("sector_decomposition"))
                     .cast<bool>()) {
                  return false;
              }
              if (!np.attr("all")(obj.attr("domain").attr("multiplicities").attr("__eq__")(1))
                     .cast<bool>()) {
                  return false;
              }
              if (!np.attr("all")(obj.attr("codomain").attr("multiplicities").attr("__eq__")(1))
                     .cast<bool>()) {
                  return false;
              }
              return true;
          }
          return py::isinstance(obj, py::module_::import("numbers").attr("Number"));
      },
      py::arg("obj"),
      doc_plus(DOC(cyten, is_scalar),
               R"pydoc(
Also accepts Python numbers and duck-typed tensors with ``domain`` / ``codomain``.
)pydoc"));

    m.def("item", &item, py::arg("tensor"), DOC(cyten, item));

    m.def(
      "linear_combination",
      [](py::object a, VectorLikeCPtr v, py::object b, VectorLikeCPtr w) {
          auto is_ok = [](py::object o) {
              if (py::isinstance(o, py::module_::import("numbers").attr("Number"))) {
                  return true;
              }
              try {
                  (void)o.cast<BlockBackend::Scalar>();
                  return true;
              } catch (py::cast_error const&) {
                  return false;
              }
          };
          if (!v || !w) {
              throw py::type_error("linear_combination() v and w must be VectorLike");
          }
          if (!is_ok(a) || !is_ok(b)) {
              throw py::type_error(
                std::format("unsupported scalar types: {}, {}",
                            std::string(py::str(py::type::of(a).attr("__name__"))),
                            std::string(py::str(py::type::of(b).attr("__name__")))));
          }
          auto sa = py::cast(v->vector_backend()->block_backend)
                      .attr("as_scalar")(a)
                      .cast<BlockBackend::Scalar>();
          auto sb = py::cast(w->vector_backend()->block_backend)
                      .attr("as_scalar")(b)
                      .cast<BlockBackend::Scalar>();
          return py_cast_vector_like(
            linear_combination(std::move(sa), std::move(v), std::move(sb), std::move(w)));
      },
      py::arg("a").none(true),
      py::arg("v"),
      py::arg("b").none(true),
      py::arg("w"),
      doc_cpp_ref(R"pydoc(The linear combination ``a * v + b * w``)pydoc",
                  "cyten::linear_combination(BlockBackend::Scalar const &, VectorLikeCPtr, "
                  "BlockBackend::Scalar const &, VectorLikeCPtr)"));

    m.def("norm",
          static_cast<BlockBackend::Scalar (*)(VectorLikeCPtr)>(&norm),
          py::arg("tensor"),
          DOC(cyten, norm, 2));

    m.def("on_device",
          &on_device,
          py::arg("tensor"),
          py::arg("device"),
          py::arg("copy") = true,
          DOC(cyten, on_device));

    m.def(
      "outer",
      [](TensorCPtr tensor1, TensorCPtr tensor2, py::object relabel1, py::object relabel2) {
          return outer(std::move(tensor1),
                       std::move(tensor2),
                       optional_relabel(relabel1),
                       optional_relabel(relabel2));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      doc_plus(DOC(cyten, outer),
               R"pydoc(
In Python, ``relabel1`` / ``relabel2`` are ``dict | None`` (``None`` = no relabel).
)pydoc"));

    m.def(
      "partial_compose",
      [](TensorCPtr tensor1,
         TensorCPtr tensor2,
         py::object tensor1_first_leg,
         py::object relabel1,
         py::object relabel2) {
          return partial_compose(std::move(tensor1),
                                 std::move(tensor2),
                                 py_as_leg_ref(tensor1_first_leg),
                                 optional_relabel(relabel1),
                                 optional_relabel(relabel2));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("tensor1_first_leg"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      doc_cpp_ref(
        R"pydoc(Tensor contraction / composition involving only a part of the full (co)domain.)pydoc",
        "cyten::partial_compose()"));

    m.def(
      "partial_trace",
      [](TensorCPtr tensor, py::args pairs, py::object levels) {
          std::vector<std::vector<LegRef>> pair_vec;
          pair_vec.reserve(static_cast<std::size_t>(pairs.size()));
          for (auto item : pairs) {
              pair_vec.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          std::optional<LevelsSpec> levels_opt;
          if (!levels.is_none()) {
              LevelsSpec spec;
              for (auto item : levels) {
                  py::object o = py::reinterpret_borrow<py::object>(item);
                  if (o.is_none()) {
                      spec.push_back(std::nullopt);
                  } else {
                      spec.push_back(o.cast<int64>());
                  }
              }
              levels_opt = std::move(spec);
          }
          return py_from_tensor_or_scalar(
            partial_trace(std::move(tensor), std::move(pair_vec), levels_opt));
      },
      py::arg("tensor"),
      py::kw_only(),
      py::arg("levels") = py::none(),
      DOC(cyten, partial_trace));

    m.def("pinv", &pinv, py::arg("tensor"), py::arg("cutoff") = 1e-15, DOC(cyten, pinv));

    m.def(
      "scalar_multiply",
      [](py::object a, py::object v) {
          if (!py::isinstance<VectorLike>(v)) {
              throw py::type_error("scalar_multiply() v must be VectorLike");
          }
          auto vec = v.cast<VectorLikeCPtr>();
          bool ok = py::isinstance(a, py::module_::import("numbers").attr("Number"));
          if (!ok) {
              try {
                  (void)a.cast<BlockBackend::Scalar>();
                  ok = true;
              } catch (py::cast_error const&) {
              }
          }
          if (!ok) {
              throw py::type_error(
                std::format("unsupported scalar type: {}",
                            std::string(py::str(py::type::of(a).attr("__name__")))));
          }
          auto s = py::cast(vec->vector_backend()->block_backend)
                     .attr("as_scalar")(a)
                     .cast<BlockBackend::Scalar>();
          return py_cast_vector_like(scalar_multiply(std::move(s), std::move(vec)));
      },
      py::arg("a").none(true),
      py::arg("v"),
      doc_cpp_ref(R"pydoc(The scalar multiplication ``a * v``)pydoc",
                  "cyten::scalar_multiply(BlockBackend::Scalar const &, VectorLikeCPtr)"));

    m.def(
      "scale_axis",
      [](TensorCPtr tensor, DiagonalTensorCPtr diag, py::object leg) {
          return scale_axis(std::move(tensor), std::move(diag), py_as_leg_ref(leg));
      },
      py::arg("tensor"),
      py::arg("diag"),
      py::arg("leg"),
      DOC(cyten, scale_axis));

    m.def(
      "tdot",
      [](TensorCPtr tensor1,
         TensorCPtr tensor2,
         py::object legs1,
         py::object legs2,
         py::object relabel1,
         py::object relabel2) {
          return py_from_tensor_or_scalar(tdot(std::move(tensor1),
                                               std::move(tensor2),
                                               py_as_leg_refs(legs1),
                                               py_as_leg_refs(legs2),
                                               optional_relabel(relabel1),
                                               optional_relabel(relabel2)));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("legs1"),
      py::arg("legs2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      doc_plus(DOC(cyten, tdot),
               R"pydoc(
In Python, ``legs1`` / ``legs2`` are ``int``, ``str``, or a sequence thereof;
``relabel1`` / ``relabel2`` are ``dict | None``.
)pydoc"));

    m.def("trace", &trace, py::arg("tensor"), DOC(cyten, trace));

    m.def("transpose", &transpose, py::arg("tensor"), DOC(cyten, transpose));
}

} // namespace cyten
