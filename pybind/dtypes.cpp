
#include <cyten/dtypes.h>
#include <pybind11/attr.h>
#include <pybind11/native_enum.h>

#include "cyten_pybind11.h"

namespace py = pybind11;
namespace cyten {

void
bind_dtypes(py::module_& m)
{
    m.def("_dtype_is_real", &dtype::is_real, py::arg("dtype"))
      .def("_dtype_is_complex", &dtype::is_complex, py::arg("dtype"))
      .def("_dtype_to_complex", &dtype::to_complex, py::arg("dtype"))
      .def("_dtype_to_real", &dtype::to_real, py::arg("dtype"))
      .def("_dtype_python_type", &dtype::python_type, py::arg("dtype"))
      .def("_dtype_zero_scalar", &dtype::zero_scalar, py::arg("dtype"))
      .def("_dtype_eps", &dtype::eps, py::arg("dtype"))
      .def("_dtype_to_numpy_dtype", &dtype::to_numpy_dtype, py::arg("dtype"))
      .def("_dtype_convert_python_scalar",
           &dtype::convert_python_scalar,
           py::arg("dtype"),
           py::arg("value"))
      .def("_dtype_from_numpy_dtype", &dtype::from_numpy_dtype, py::arg("dtype"))
      .def("_dtype_common", &dtype::common, py::arg("dtypes"));

    py::native_enum<Dtype> dtype_enum(m,
                                      "Dtype",
                                      "cyten.block_backends.dtypes._DtypeEnumWrapper",
                                      R"pydoc(
        The dtype of (entries in) a tensor.

        value = num_bytes * 2 + int(not is_real)
        )pydoc");
    dtype_enum.value("bool", Dtype::Bool)
      .value("float32", Dtype::Float32)
      .value("complex64", Dtype::Complex64)
      .value("float64", Dtype::Float64)
      .value("complex128", Dtype::Complex128)
      .export_values()
      .finalize();
}

} // namespace cyten
