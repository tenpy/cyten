#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/mask.h>

#include "py_callbacks.hpp"
#include "py_factory_parse.hpp"

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include "docstrings/tensors/mask.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

namespace {

std::optional<std::vector<std::variant<int64, std::string>>>
optional_leg_order(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<std::variant<int64, std::string>> out;
    for (auto item : to_iterable(obj)) {
        if (py::isinstance<py::str>(item)) {
            out.emplace_back(item.cast<std::string>());
        } else {
            out.emplace_back(item.cast<int64>());
        }
    }
    return out;
}

} // namespace

void
bind_tensors_mask(py::module_& m)
{
    py::class_<Mask, Tensor, py::smart_holder> cls(m, "Mask");
    cls.doc() = DOC(cyten, Mask);

    cls.def(py::init([](TensorBackend::DataPtr data,
                        py::object space_in_obj,
                        py::object space_out_obj,
                        std::optional<bool> is_projection,
                        TensorBackend::Ptr backend,
                        py::object labels) {
                auto space_in = py_as_space_leg(space_in_obj);
                auto space_out = py_as_space_leg(space_out_obj);
                bool proj = false;
                if (!is_projection.has_value()) {
                    if (space_in->Space::dim == space_out->Space::dim) {
                        throw std::invalid_argument(
                          "Need to specify is_projection for equal spaces.");
                    }
                    proj = space_in->Space::dim > space_out->Space::dim;
                } else {
                    proj = *is_projection;
                }
                auto init = parse_tensor_init(py::make_tuple(space_out_obj),
                                              py::make_tuple(space_in_obj),
                                              std::move(backend),
                                              labels);
                auto device_s = init.backend->get_device_from_data(data);
                return std::make_shared<Mask>(std::move(data),
                                              space_in,
                                              space_out,
                                              proj,
                                              init.backend,
                                              init.symmetry,
                                              init.labels,
                                              std::move(device_s));
            }),
            py::arg("data"),
            py::arg("space_in"),
            py::arg("space_out"),
            py::arg("is_projection") = py::none(),
            py::arg("backend") = nullptr,
            py::arg("labels") = py::none());

    cls.def_readonly_static("_forbidden_dtypes", &Mask::_forbidden_dtypes);

    cls.def_readwrite("is_projection", &Mask::is_projection);

    cls.def_property(
      "data",
      [](Mask& self) -> py::object {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](Mask& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data = NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
          } else {
              self.data = obj.cast<TensorBackend::DataPtr>();
          }
      });

    cls.def_property_readonly("large_leg", &Mask::large_leg);
    cls.def_property_readonly("small_leg", &Mask::small_leg);

    cls.def("test_sanity",
            &Mask::test_sanity,
            DOC(cyten, Mask, test_sanity));

    cls.def_static(
      "from_eye",
      [](py::object leg,
         bool is_projection,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(leg, std::move(backend), labels);
          return Mask::from_eye(
            py_as_space_leg(leg), is_projection, init.backend, init.labels, device);
      },
      py::arg("leg"),
      py::arg("is_projection") = true,
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      DOC(cyten, Mask, from_eye));

    cls.def_static(
      "from_block_mask",
      [](py::object block_mask,
         py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          auto block = init.backend->block_backend->as_block(block_mask, Dtype::Bool, device);
          return Mask::from_block_mask(
            block, py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      py::arg("block_mask"),
      py::arg("large_leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      DOC(cyten, Mask, from_block_mask));

    cls.def_static(
      "from_DiagonalTensor",
      [](py::object diag) { return Mask::from_DiagonalTensor(diag.cast<DiagonalTensorCPtr>()); },
      py::arg("diag"),
      DOC(cyten, Mask, from_DiagonalTensor));

    cls.def_static(
      "from_indices",
      [](py::object indices,
         py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          return Mask::from_indices(
            indices, py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      py::arg("indices"),
      py::arg("large_leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      DOC(cyten, Mask, from_indices));

    cls.def_static(
      "from_random",
      [](py::object large_leg,
         py::object small_leg,
         TensorBackend::Ptr backend,
         float64 p_keep,
         int64 min_keep,
         py::object labels,
         std::optional<std::string> device,
         py::object np_random) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          Space::Ptr small;
          if (!small_leg.is_none()) {
              small = py_as_space_leg(small_leg);
          }
          return Mask::from_random(py_as_space_leg(large_leg),
                                   small,
                                   init.backend,
                                   p_keep,
                                   min_keep,
                                   init.labels,
                                   device,
                                   np_random);
      },
      py::arg("large_leg"),
      py::arg("small_leg") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("p_keep") = 0.5,
      py::arg("min_keep") = 0,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      py::arg("np_random") = py::none(),
      DOC(cyten, Mask, from_random));

    cls.def_static(
      "from_zero",
      [](py::object large_leg,
         TensorBackend::Ptr backend,
         py::object labels,
         std::optional<std::string> device) {
          auto init = py_parse_diag(large_leg, std::move(backend), labels);
          return Mask::from_zero(py_as_space_leg(large_leg), init.backend, init.labels, device);
      },
      py::arg("large_leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("device") = py::none(),
      DOC(cyten, Mask, from_zero));

    cls.def_static("from_hdf5",
                   &Mask::from_hdf5,
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   DOC(cyten, Mask, from_hdf5));

    cls.def("save_hdf5",
            &Mask::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            DOC(cyten, Mask, save_hdf5));

    cls.def("as_dtype",
            &Mask::as_dtype,
            py::arg("dtype"),
            DOC(cyten, Mask, as_dtype));

    cls.def(
      "as_SymmetricTensor",
      [](Mask& self,
         bool guarantee_copy,
         std::optional<std::string> warning,
         std::optional<Dtype> dtype) {
          if (dtype.has_value()) {
              return self.as_SymmetricTensor(guarantee_copy, warning, *dtype);
          }
          return self.as_SymmetricTensor(guarantee_copy, warning);
      },
      py::arg("guarantee_copy") = false,
      py::arg("warning") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      DOC(cyten, Mask, as_SymmetricTensor));

    cls.def("as_DiagonalTensor", &Mask::as_DiagonalTensor, py::arg("dtype") = Dtype::Complex128);

    cls.def("as_block_mask", &Mask::as_block_mask);
    cls.def("as_numpy_mask", &Mask::as_numpy_mask);

    cls.def("all",
            &Mask::all,
            DOC(cyten, Mask, all));
    cls.def("any",
            &Mask::any,
            DOC(cyten, Mask, any));

    cls.def("copy",
            &Mask::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none(),
            DOC(cyten, Mask, copy));

    // Override Tensor.dagger / hc properties (which delegate to the Python free function).
    cls.def_property_readonly("dagger",
                              &Mask::dagger,
                              DOC(cyten, Mask, dagger));
    cls.def_property_readonly("hc",
                              &Mask::dagger,
                              DOC(cyten, Mask, dagger));

    cls.def("_get_item",
            &Mask::_get_item,
            py::arg("idx"),
            DOC(cyten, Mask, _get_item));

    cls.def("logical_not",
            &Mask::logical_not,
            DOC(cyten, Mask, logical_not));
    cls.def("orthogonal_complement",
            &Mask::orthogonal_complement,
            DOC(cyten, Mask, orthogonal_complement));

    cls.def("move_to_device",
            &Mask::move_to_device,
            py::arg("device"),
            DOC(cyten, Mask, move_to_device));

    cls.def("to_backend",
            &Mask::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none(),
            DOC(cyten, Mask, to_backend));

    cls.def(
      "to_dense_block",
      [](Mask& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(cyten, Mask, to_dense_block));

    cls.def(
      "to_numpy",
      [](Mask& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
          return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("numpy_dtype") = py::none(),
      py::arg("understood_braiding") = false,
      DOC(cyten, Mask, to_numpy));

    cls.def(
      "_binary_operand",
      [](Mask& self,
         py::object other,
         py::function func,
         std::string operand,
         bool return_NotImplemented) -> py::object {
          auto adapted = adapt_block_bool_binary(func, self.backend->block_backend);
          if (py::isinstance<py::bool_>(other)) {
              return py::cast(self._binary_operand(other.cast<bool>(), adapted, operand));
          }
          if (py::isinstance<Mask>(other) ||
              py::isinstance(other, py::module_::import("cyten.tensors._tensors").attr("Mask"))) {
              return py::cast(self._binary_operand(other.cast<MaskCPtr>(), adapted, operand));
          }
          if (return_NotImplemented &&
              !(py::isinstance<Tensor>(other) ||
                py::isinstance(other,
                               py::module_::import("cyten.tensors._tensors").attr("Tensor")) ||
                py::isinstance(other, py::module_::import("numbers").attr("Number")))) {
              return py::reinterpret_borrow<py::object>(Py_NotImplemented);
          }
          throw std::invalid_argument(std::format("Invalid types for operand \"{}\": Mask and {}",
                                                  operand,
                                                  std::string(py::str(py::type::of(other)))));
      },
      py::arg("other"),
      py::arg("func"),
      py::arg("operand"),
      py::arg("return_NotImplemented") = true,
      DOC(cyten, Mask, _binary_operand));

    cls.def(
      "_unary_operand",
      [](Mask& self, py::function func) {
          return self._unary_operand(adapt_block_bool_unary(func, self.backend->block_backend));
      },
      py::arg("func"));

    cls.def("__bool__", [](Mask&) {
        throw py::type_error("The truth value of a Mask is ambiguous. Use a.any() or a.all()");
    });

    cls.def("__invert__", [](Mask& self) {
        return self._unary_operand(adapt_block_bool_unary(
          py::module_::import("operator").attr("invert"), self.backend->block_backend));
    });

    auto bind_bool_binop = [&](char const* name, char const* op_name, char const* operand) {
        cls.def(
          name,
          [op_name, operand](Mask& self, py::object other) -> py::object {
              auto func = adapt_block_bool_binary(py::module_::import("operator").attr(op_name),
                                                  self.backend->block_backend);
              if (py::isinstance<py::bool_>(other)) {
                  return py::cast(self._binary_operand(other.cast<bool>(), func, operand));
              }
              if (py::isinstance<Mask>(other) ||
                  py::isinstance(other,
                                 py::module_::import("cyten.tensors._tensors").attr("Mask"))) {
                  return py::cast(self._binary_operand(other.cast<MaskCPtr>(), func, operand));
              }
              if (!(py::isinstance<Tensor>(other) ||
                    py::isinstance(other,
                                   py::module_::import("cyten.tensors._tensors").attr("Tensor")) ||
                    py::isinstance(other, py::module_::import("numbers").attr("Number")))) {
                  return py::reinterpret_borrow<py::object>(Py_NotImplemented);
              }
              throw std::invalid_argument(
                std::format("Invalid types for operand \"{}\": Mask and {}",
                            operand,
                            std::string(py::str(py::type::of(other)))));
          },
          py::arg("other"));
    };
    bind_bool_binop("__and__", "and_", "&");
    bind_bool_binop("__rand__", "and_", "&");
    bind_bool_binop("__or__", "or_", "|");
    bind_bool_binop("__ror__", "or_", "|");
    bind_bool_binop("__xor__", "xor", "^");
    bind_bool_binop("__rxor__", "xor", "^");
    bind_bool_binop("__eq__", "eq", "==");
    bind_bool_binop("__ne__", "ne", "!=");
}

} // namespace cyten
