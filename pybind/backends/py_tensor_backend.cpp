#include "../py_cyten_pybind11.h"
#include "../doc_plus.h"
#include "docstrings/backends/tensor_backend.h"
#include "../tensors/py_callbacks.hpp"
#include "py_trampolines.hpp"

#include <cyten/backends/tensor_backend.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/symmetric_tensor.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

void
bind_tensor_backend(py::module_& m)
{
    py::class_<TensorBackend::Data, py::smart_holder> data_cls(m, "TensorBackendData");
    data_cls.doc() = "Backend-specific payload stored on a tensor (except symmetry data on legs).";

    py::class_<TensorBackend, PyTensorBackend, py::smart_holder> tensor_backend(m,
                                                                                "TensorBackend");
    tensor_backend.doc() = DOC(cyten, TensorBackend);

    tensor_backend.def(py::init<std::shared_ptr<BlockBackend>>(), py::arg("block_backend"))
      .def_property_readonly("can_decompose_tensors", &TensorBackend::can_decompose_tensors)
      .def_readwrite("block_backend", &TensorBackend::block_backend);

    tensor_backend //  methods
      .def("__repr__", &TensorBackend::__repr__)
      .def("__str__", &TensorBackend::__str__)
      .def("__eq__",
           [](TensorBackend const& self, py::object other) {
               if (!py::isinstance<TensorBackend>(other)) {
                   return false;
               }
               return self == other.cast<TensorBackend const&>();
           })
      .def("item",
           &TensorBackend::item,
           py::arg("a"),
           DOC(cyten, TensorBackend, item))
      .def("test_tensor_sanity",
           &TensorBackend::test_tensor_sanity,
           py::arg("a"),
           py::arg("is_diagonal"),
           DOC(cyten, TensorBackend, test_tensor_sanity))
      .def("test_mask_sanity", &TensorBackend::test_mask_sanity, py::arg("a"))
      .def("make_pipe",
           &TensorBackend::make_pipe,
           py::arg("legs"),
           py::arg("is_dual"),
           py::arg("pipe") = py::none(),
           DOC(cyten, TensorBackend, make_pipe))
      .def(
        "act_block_diagonal_square_matrix",
        [](TensorBackend& self,
           SymmetricTensorCPtr a,
           py::function block_method,
           py::object dtype_map) {
            return self.act_block_diagonal_square_matrix(
              a, block_unary_from_python(block_method), dtype_map_from_python(dtype_map));
        },
        py::arg("a"),
        py::arg("block_method"),
        py::arg("dtype_map"),
        DOC(cyten, TensorBackend, act_block_diagonal_square_matrix))
      .def("add_trivial_leg",
           &TensorBackend::add_trivial_leg,
           py::arg("a"),
           py::arg("legs_pos"),
           py::arg("add_to_domain"),
           py::arg("co_domain_pos"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           DOC(cyten, TensorBackend, add_trivial_leg))
      .def("almost_equal",
           &TensorBackend::almost_equal,
           py::arg("a"),
           py::arg("b"),
           py::arg("rtol"),
           py::arg("atol"),
           DOC(cyten, TensorBackend, almost_equal))
      .def("apply_mask_to_DiagonalTensor",
           &TensorBackend::apply_mask_to_DiagonalTensor,
           py::arg("tensor"),
           py::arg("mask"))
      .def("combine_legs",
           &TensorBackend::combine_legs,
           py::arg("tensor"),
           py::arg("leg_idcs_combine"),
           py::arg("pipes"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           DOC(cyten, TensorBackend, combine_legs))
      .def("compose",
           &TensorBackend::compose,
           py::arg("a"),
           py::arg("b"),
           DOC(cyten, TensorBackend, compose))
      .def("copy_data",
           &TensorBackend::copy_data,
           py::arg("a"),
           py::arg("device") = py::none(),
           DOC(cyten, TensorBackend, copy_data))
      .def("dagger",
           &TensorBackend::dagger,
           py::arg("a"),
           DOC(cyten, TensorBackend, dagger))
      .def("data_item",
           &TensorBackend::data_item,
           py::arg("a"),
           DOC(cyten, TensorBackend, data_item))
      .def("diagonal_all",
           &TensorBackend::diagonal_all,
           py::arg("a"),
           DOC(cyten, TensorBackend, diagonal_all))
      .def("diagonal_any",
           &TensorBackend::diagonal_any,
           py::arg("a"),
           DOC(cyten, TensorBackend, diagonal_any))
      .def(
        "diagonal_elementwise_binary",
        [](TensorBackend& self,
           DiagonalTensorCPtr a,
           DiagonalTensorCPtr b,
           py::function func,
           py::dict func_kwargs,
           bool partial_zero_is_zero) {
            return self.diagonal_elementwise_binary(
              a, b, block_binary_from_python(func, func_kwargs), partial_zero_is_zero);
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("func"),
        py::arg("func_kwargs"),
        py::arg("partial_zero_is_zero"),
        DOC(cyten, TensorBackend, diagonal_elementwise_binary))
      .def(
        "diagonal_elementwise_unary",
        [](TensorBackend& self,
           DiagonalTensorCPtr a,
           py::function func,
           py::dict func_kwargs,
           bool maps_zero_to_zero) {
            return self.diagonal_elementwise_unary(
              a, block_unary_from_python(func, func_kwargs), maps_zero_to_zero);
        },
        py::arg("a"),
        py::arg("func"),
        py::arg("func_kwargs"),
        py::arg("maps_zero_to_zero"),
        DOC(cyten, TensorBackend, diagonal_elementwise_unary))
      .def("diagonal_from_block",
           &TensorBackend::diagonal_from_block,
           py::arg("a"),
           py::arg("co_domain"),
           py::arg("tol"),
           DOC(cyten, TensorBackend, diagonal_from_block))
      .def(
        "diagonal_from_sector_block_func",
        [](TensorBackend& self, py::function func, TensorProduct::Ptr co_domain) {
            return self.diagonal_from_sector_block_func(sector_block_factory_from_python(func),
                                                        std::move(co_domain));
        },
        py::arg("func"),
        py::arg("co_domain"),
        DOC(cyten, TensorBackend, diagonal_from_sector_block_func))
      .def("diagonal_tensor_from_full_tensor",
           &TensorBackend::diagonal_tensor_from_full_tensor,
           py::arg("a"),
           py::arg("tol") = 1e-12,
           DOC(cyten, TensorBackend, diagonal_tensor_from_full_tensor))
      .def("diagonal_tensor_trace_full", &TensorBackend::diagonal_tensor_trace_full, py::arg("a"))
      .def("diagonal_tensor_to_block",
           &TensorBackend::diagonal_tensor_to_block,
           py::arg("a"),
           DOC(cyten, TensorBackend, diagonal_tensor_to_block))
      .def("diagonal_to_mask",
           &TensorBackend::diagonal_to_mask,
           py::arg("tens"),
           DOC(cyten, TensorBackend, diagonal_to_mask))
      .def("diagonal_transpose",
           &TensorBackend::diagonal_transpose,
           py::arg("tens"),
           DOC(cyten, TensorBackend, diagonal_transpose))
      .def("eigh",
           &TensorBackend::eigh,
           py::arg("a"),
           py::arg("new_leg_dual"),
           py::arg("sort") = py::none(),
           DOC(cyten, TensorBackend, eigh))
      .def("eye_data",
           &TensorBackend::eye_data,
           py::arg("co_domain"),
           py::arg("dtype"),
           py::arg("device"),
           DOC(cyten, TensorBackend, eye_data))
      .def("from_dense_block",
           &TensorBackend::from_dense_block,
           py::arg("a"),
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("tol"),
           DOC(cyten, TensorBackend, from_dense_block))
      .def("from_dense_block_trivial_sector",
           &TensorBackend::from_dense_block_trivial_sector,
           py::arg("block"),
           py::arg("leg"),
           DOC(cyten, TensorBackend, from_dense_block_trivial_sector))
      .def("from_grid",
           &TensorBackend::from_grid,
           py::arg("grid"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           py::arg("left_mult_slices"),
           py::arg("right_mult_slices"),
           py::arg("dtype"),
           py::arg("device"),
           DOC(cyten, TensorBackend, from_grid))
      .def("from_random_normal",
           &TensorBackend::from_random_normal,
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("sigma"),
           py::arg("dtype"),
           py::arg("device"))
      .def(
        "from_sector_block_func",
        [](TensorBackend& self,
           py::function func,
           TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain) {
            return self.from_sector_block_func(
              sector_block_factory_from_python(func), std::move(codomain), std::move(domain));
        },
        py::arg("func"),
        py::arg("codomain"),
        py::arg("domain"),
        DOC(cyten, TensorBackend, from_sector_block_func))
      .def("from_tree_pairs",
           &TensorBackend::from_tree_pairs,
           py::arg("trees"),
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("dtype"),
           py::arg("device"),
           DOC(cyten, TensorBackend, from_tree_pairs))
      .def("full_data_from_diagonal_tensor",
           &TensorBackend::full_data_from_diagonal_tensor,
           py::arg("a"))
      .def("full_data_from_mask",
           &TensorBackend::full_data_from_mask,
           py::arg("a"),
           py::arg("dtype"),
           DOC(cyten, TensorBackend, full_data_from_mask))
      .def("get_device_from_data",
           &TensorBackend::get_device_from_data,
           py::arg("a"),
           DOC(cyten, TensorBackend, get_device_from_data))
      .def("get_dtype_from_data", &TensorBackend::get_dtype_from_data, py::arg("a"))
      .def("get_element",
           &TensorBackend::get_element,
           py::arg("a"),
           py::arg("idcs"),
           DOC(cyten, TensorBackend, get_element))
      .def("get_element_diagonal",
           &TensorBackend::get_element_diagonal,
           py::arg("a"),
           py::arg("idx"),
           DOC(cyten, TensorBackend, get_element_diagonal))
      .def("get_element_mask",
           &TensorBackend::get_element_mask,
           py::arg("a"),
           py::arg("idcs"),
           DOC(cyten, TensorBackend, get_element_mask))
      .def("inner",
           &TensorBackend::inner,
           py::arg("a"),
           py::arg("b"),
           py::arg("do_dagger"),
           DOC(cyten, TensorBackend, inner))
      .def("inv_part_from_dense_block_single_sector",
           &TensorBackend::inv_part_from_dense_block_single_sector,
           py::arg("vector"),
           py::arg("space"),
           py::arg("charge_leg"),
           DOC(cyten, TensorBackend, inv_part_from_dense_block_single_sector))
      .def("inv_part_to_dense_block_single_sector",
           &TensorBackend::inv_part_to_dense_block_single_sector,
           py::arg("tensor"),
           DOC(cyten, TensorBackend, inv_part_to_dense_block_single_sector))
      .def("linear_combination",
           &TensorBackend::linear_combination,
           py::arg("a"),
           py::arg("v"),
           py::arg("b"),
           py::arg("w"),
           DOC(cyten, TensorBackend, linear_combination))
      .def("lq",
           &TensorBackend::lq,
           py::arg("tensor"),
           py::arg("new_co_domain"),
           DOC(cyten, TensorBackend, lq))
      .def(
        "mask_binary_operand",
        [](TensorBackend& self, MaskCPtr mask1, MaskCPtr mask2, py::function func) {
            return self.mask_binary_operand(mask1, mask2, block_binary_from_python(func));
        },
        py::arg("mask1"),
        py::arg("mask2"),
        py::arg("func"),
        DOC(cyten, TensorBackend, mask_binary_operand))
      .def("mask_contract_large_leg",
           &TensorBackend::mask_contract_large_leg,
           py::arg("tensor"),
           py::arg("mask"),
           py::arg("leg_idx"),
           DOC(cyten, TensorBackend, mask_contract_large_leg))
      .def("mask_contract_small_leg",
           &TensorBackend::mask_contract_small_leg,
           py::arg("tensor"),
           py::arg("mask"),
           py::arg("leg_idx"),
           DOC(cyten, TensorBackend, mask_contract_small_leg))
      .def("mask_dagger", &TensorBackend::mask_dagger, py::arg("mask"))
      .def("mask_from_block",
           &TensorBackend::mask_from_block,
           py::arg("a"),
           py::arg("large_leg"),
           DOC(cyten, TensorBackend, mask_from_block))
      .def("mask_to_block",
           &TensorBackend::mask_to_block,
           py::arg("a"),
           DOC(cyten, TensorBackend, mask_to_block))
      .def("mask_to_diagonal", &TensorBackend::mask_to_diagonal, py::arg("a"), py::arg("dtype"))
      .def("mask_transpose",
           &TensorBackend::mask_transpose,
           py::arg("tens"),
           DOC(cyten, TensorBackend, mask_transpose))
      .def(
        "mask_unary_operand",
        [](TensorBackend& self, MaskCPtr mask, py::function func) {
            return self.mask_unary_operand(mask, block_unary_from_python(func));
        },
        py::arg("mask"),
        py::arg("func"),
        DOC(cyten, TensorBackend, mask_unary_operand))
      .def("move_to_device",
           &TensorBackend::move_to_device,
           py::arg("a"),
           py::arg("device"),
           DOC(cyten, TensorBackend, move_to_device))
      .def("mul", &TensorBackend::mul, py::arg("a"), py::arg("b"))
      .def("norm",
           &TensorBackend::norm,
           py::arg("a"),
           DOC(cyten, TensorBackend, norm))
      .def("outer",
           &TensorBackend::outer,
           py::arg("a"),
           py::arg("b"),
           DOC(cyten, TensorBackend, outer))
      .def("partial_compose",
           &TensorBackend::partial_compose,
           py::arg("a"),
           py::arg("b"),
           py::arg("a_first_leg"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           DOC(cyten, TensorBackend, partial_compose))
      .def("partial_trace",
           &TensorBackend::partial_trace,
           py::arg("tensor"),
           py::arg("pairs"),
           py::arg("levels"),
           DOC(cyten, TensorBackend, partial_trace))
      .def("permute_legs",
           &TensorBackend::permute_legs,
           py::arg("a"),
           py::arg("codomain_idcs"),
           py::arg("domain_idcs"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           py::arg("mixes_codomain_domain"),
           py::arg("levels"),
           py::arg("bend_right"),
           DOC(cyten, TensorBackend, permute_legs))
      .def("qr",
           &TensorBackend::qr,
           py::arg("a"),
           py::arg("new_co_domain"),
           DOC(cyten, TensorBackend, qr))
      .def(
        "reduce_DiagonalTensor",
        [](TensorBackend& self,
           DiagonalTensorCPtr tensor,
           py::function block_func,
           py::function func) {
            return self.reduce_DiagonalTensor(
              tensor, block_to_scalar_from_python(block_func), scalar_reduce_from_python(func));
        },
        py::arg("tensor"),
        py::arg("block_func"),
        py::arg("func"),
        DOC(cyten, TensorBackend, reduce_DiagonalTensor))
      .def("scale_axis",
           &TensorBackend::scale_axis,
           py::arg("a"),
           py::arg("b"),
           py::arg("leg"),
           DOC(cyten, TensorBackend, scale_axis))
      .def("split_legs",
           &TensorBackend::split_legs,
           py::arg("a"),
           py::arg("leg_idcs"),
           py::arg("new_codomain"),
           py::arg("new_domain"),
           DOC(cyten, TensorBackend, split_legs))
      .def("squeeze_legs",
           &TensorBackend::squeeze_legs,
           py::arg("a"),
           py::arg("idcs"),
           DOC(cyten, TensorBackend, squeeze_legs))
      .def("supports_symmetry", &TensorBackend::supports_symmetry, py::arg("symmetry"))
      .def("svd",
           &TensorBackend::svd,
           py::arg("a"),
           py::arg("new_co_domain"),
           py::arg("algorithm"),
           DOC(cyten, TensorBackend, svd))
      .def("state_tensor_product",
           &TensorBackend::state_tensor_product,
           py::arg("state1"),
           py::arg("state2"),
           py::arg("pipe"),
           DOC(cyten, TensorBackend, state_tensor_product))
      .def("to_block_backend",
           &TensorBackend::to_block_backend,
           py::arg("data"),
           py::arg("block_backend"),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none())
      .def("to_dense_block",
           &TensorBackend::to_dense_block,
           py::arg("a"),
           DOC(cyten, TensorBackend, to_dense_block))
      .def("to_dense_block_trivial_sector",
           &TensorBackend::to_dense_block_trivial_sector,
           py::arg("tensor"),
           DOC(cyten, TensorBackend, to_dense_block_trivial_sector))
      .def("to_dtype",
           &TensorBackend::to_dtype,
           py::arg("a"),
           py::arg("dtype"),
           DOC(cyten, TensorBackend, to_dtype))
      .def("trace_full",
           &TensorBackend::trace_full,
           py::arg("a"),
           py::arg("idcs1") = std::vector<int64>{},
           py::arg("idcs2") = std::vector<int64>{})
      .def("truncate_singular_values",
           &TensorBackend::truncate_singular_values,
           py::arg("S"),
           py::arg("chi_max"),
           py::arg("chi_min"),
           py::arg("degeneracy_tol"),
           py::arg("trunc_cut"),
           py::arg("svd_min"),
           py::arg("minimize_error") = true,
           DOC(cyten, TensorBackend, truncate_singular_values))
      .def("_truncate_singular_values_selection",
           &TensorBackend::_truncate_singular_values_selection,
           py::arg("S"),
           py::arg("qdims"),
           py::arg("chi_max"),
           py::arg("chi_min"),
           py::arg("degeneracy_tol"),
           py::arg("trunc_cut"),
           py::arg("svd_min"),
           py::arg("minimize_error") = true,
           DOC(cyten, TensorBackend, _truncate_singular_values_selection))
      .def("zero_data",
           &TensorBackend::zero_data,
           py::arg("codomain"),
           py::arg("domain"),
           py::arg("dtype"),
           py::arg("device"),
           py::arg("all_blocks") = false,
           DOC(cyten, TensorBackend, zero_data))
      .def("zero_diagonal_data",
           &TensorBackend::zero_diagonal_data,
           py::arg("co_domain"),
           py::arg("dtype"),
           py::arg("device"))
      .def(
        "zero_mask_data", &TensorBackend::zero_mask_data, py::arg("large_leg"), py::arg("device"))
      .def("is_real",
           &TensorBackend::is_real,
           py::arg("a"),
           DOC(cyten, TensorBackend, is_real))
      .def("save_hdf5",
           &TensorBackend::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"));

    py::object classmethod = py::module_::import("builtins").attr("classmethod");
    tensor_backend.attr("from_hdf5") = classmethod(
      py::cpp_function(&TensorBackend::from_hdf5,
                       py::name("from_hdf5"),
                       py::arg("cls"),
                       py::arg("hdf5_loader"),
                       py::arg("h5gr"),
                       py::arg("subpath"),
                       "Reconstruct a tensor backend from HDF5 by loading its BlockBackend."));

    // Nested Data type under TensorBackend for Python parity with BlockBackend.BlockCls style.
    tensor_backend.attr("Data") = data_cls;

    m.def("conventional_leg_order",
          py::overload_cast<py::object, py::object>(&conventional_leg_order),
          py::arg("tensor_or_codomain"),
          py::arg("domain") = py::none(),
          doc_cpp_ref(R"pydoc(conventional_leg_order)pydoc", "cyten::TensorBackend::conventional_leg_order()"));

    m.def(
      "get_same_backend",
      [](py::args objs, py::kwargs kwargs) {
          std::string error_msg = "Incompatible backends.";
          if (kwargs.contains("error_msg"))
              error_msg = kwargs["error_msg"].cast<std::string>();
          std::vector<py::object> vec;
          vec.reserve(objs.size());
          for (auto const& o : objs)
              vec.emplace_back(py::reinterpret_borrow<py::object>(o));
          return get_same_backend(vec, std::move(error_msg));
      },
      doc_cpp_ref(R"pydoc(get_same_backend)pydoc", "cyten::TensorBackend::get_same_backend()"));
}

} // namespace cyten
