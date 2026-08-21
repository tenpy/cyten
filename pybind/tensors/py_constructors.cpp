#include <cyten/tensors/constructors.h>
#include <cyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"
#include "py_factory_parse.hpp"

#include "docstrings/tensors/constructors.h"

#include <optional>
#include <string>
#include <vector>

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
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return eye(py_as_space_leg(leg),
                     init.backend,
                     init.labels,
                     dtype,
                     std::move(device_opt),
                     diagonal);
      },
      py::arg("leg"),
      py::arg("backend") = py::none(),
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Float64,
      py::arg("device") = py::none(),
      py::arg("diagonal") = true,
      DOC(cyten, eye));

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
          if (py::isinstance<Tensor>(obj)) {
              auto t = obj.cast<TensorCPtr>();
              auto cod = tensor_product_from_python(codomain, t->symmetry);
              TensorProduct::Ptr dom;
              if (!domain.is_none()) {
                  dom = tensor_product_from_python(domain, t->symmetry);
              }
              std::optional<LegLabels> labs;
              if (!labels.is_none()) {
                  labs = parse_tensor_init_labels(labels, t->codomain, t->domain);
              }
              return py::cast(tensor(t,
                                     std::move(cod),
                                     std::move(dom),
                                     std::move(backend),
                                     labs,
                                     dtype_opt,
                                     device_opt));
          }
          auto init = parse_tensor_init(codomain, domain, std::move(backend), labels);
          auto block = init.backend->block_backend->as_block(obj, dtype_opt, device_opt);
          return py::cast(tensor(block,
                                 init.codomain,
                                 init.domain,
                                 init.backend,
                                 init.labels,
                                 dtype_opt,
                                 device_opt,
                                 understood_braiding));
      },
      py::arg("obj"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = py::none(),
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("understood_braiding") = false,
      doc_plus(DOC(cyten, tensor),
               R"pydoc(
Also accepts a dense block (second C++ overload). ``None`` for optional arguments
matches C++ null / ``nullopt``. ``understood_braiding`` applies only to the block path.
)pydoc"));

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
          return add_trivial_leg(tens.cast<TensorCPtr>(),
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
      DOC(cyten, add_trivial_leg));

    m.def(
      "zero_like",
      [](py::object tensor) { return zero_like(tensor.cast<TensorCPtr>()); },
      py::arg("tensor"),
      DOC(cyten, zero_like));

    m.def(
      "tensor_from_grid",
      [](py::object grid, py::object labels, py::object dtype) {
          std::optional<Dtype> dtype_opt;
          if (!dtype.is_none()) {
              dtype_opt = dtype.cast<Dtype>();
          }
          std::vector<std::vector<TensorPtr>> g;
          for (auto row_h : py::reinterpret_borrow<py::iterable>(grid)) {
              std::vector<TensorPtr> row;
              for (auto item : py::reinterpret_borrow<py::iterable>(row_h)) {
                  py::object obj = py::reinterpret_borrow<py::object>(item);
                  row.push_back(obj.is_none() ? nullptr : obj.cast<TensorPtr>());
              }
              g.push_back(std::move(row));
          }
          auto res = tensor_from_grid(std::move(g), std::nullopt, dtype_opt);
          if (!labels.is_none()) {
              res->set_labels(parse_tensor_init_labels(labels, res->codomain, res->domain));
          }
          return res;
      },
      py::arg("grid"),
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      doc_plus(DOC(cyten, tensor_from_grid),
               R"pydoc(
In Python, ``grid`` is ``list[list[SymmetricTensor | None]]`` (``None`` = zero cell);
``labels`` / ``dtype`` use ``None`` for C++ ``nullopt``.
)pydoc"));
}

} // namespace cyten
