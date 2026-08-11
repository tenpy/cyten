#pragma once

#include <cyten/cyten.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cyten {

namespace py = pybind11;

// the _core.cpp file implements the main pybind11 python module cyten._core with all python
// bindings.

// here, we have declarations of binding functions defined in the corresponding *.cpp files.

void bind_version(py::module_& m);
void bind_config(py::module_& m);
void bind_tools(py::module_& m);
void bind_cost_polynomials(py::module_& m);
void bind_mappings(py::module_& m);
void bind_block_backend(py::module_& m);
void bind_symmetries(py::module_& m);
void bind_tensor_backend(py::module_& m);
void bind_no_symmetry_backend(py::module_& m);
void bind_block_inds(py::module_& m);
void bind_abelian_backend_data(py::module_& m);
void bind_abelian_backend(py::module_& m);
void bind_fusion_tree_data(py::module_& m);
void bind_fusion_tree_backend(py::module_& m);
void bind_fusion_tree_mapping(py::module_& m);
void bind_backend_factory(py::module_& m);
void bind_tensors_labels(py::module_& m);
void bind_tensors_tensor(py::module_& m);
void bind_tensors_symmetric_tensor(py::module_& m);
void bind_tensors_diagonal_tensor(py::module_& m);
void bind_tensors_mask(py::module_& m);
void bind_tensors_charged_tensor(py::module_& m);
void bind_tensors_helpers(py::module_& m);
void bind_tensors_constructors(py::module_& m);
void bind_tensors_ops_elementwise(py::module_& m);
void bind_tensors_ops_algebra(py::module_& m);
void bind_tensors_ops_legs(py::module_& m);
void bind_check(py::module_& m);

} // namespace cyten
