// NOTE: this file is #included from block_backend.cpp

#include <cyten/block_backend/dtypes.h>
#include <pybind11/attr.h>
#include <pybind11/native_enum.h>

#include <string>
#include <type_traits>
#include <vector>

namespace cyten {

void
bind_block_backend_dtypes(py::module_& m)
{
    py::native_enum<Dtype> dtype_enum(m,
                                      "Dtype",
                                      "enum.Enum",
                                      R"pydoc(
                                      The dtype of (entries in) a tensor.

                                      value = num_bytes * 2 + int(not is_real)
                                      )pydoc");
    dtype_enum.value("bool", Dtype::Bool)
      .value("float32", Dtype::Float32)
      .value("complex64", Dtype::Complex64)
      .value("float64", Dtype::Float64)
      .value("complex128", Dtype::Complex128)
      .value("int64", Dtype::Int64)
      .export_values()
      .finalize();

    // native_enum has no .def(); attach methods/properties after finalize.
    py::object D = m.attr("Dtype");
    py::object property = py::module_::import("builtins").attr("property");
    py::object classmethod = py::module_::import("builtins").attr("classmethod");

    auto as_property = [&](const char* name, auto&& fn) {
        D.attr(name) = property(
          py::cpp_function(std::forward<decltype(fn)>(fn), py::name(name), py::is_method(D)));
    };

    as_property("is_real", &dtype::is_real);
    as_property("is_complex", &dtype::is_complex);
    as_property("to_complex", &dtype::to_complex);
    as_property("to_real", &dtype::to_real);
    as_property("python_type", &dtype::python_type);
    as_property("zero_scalar", &dtype::zero_scalar);
    as_property("one_scalar", &dtype::one_scalar);
    as_property("eps", &dtype::eps);

    // Methods

    D.attr("to_numpy_dtype") =
      py::cpp_function(&dtype::to_numpy_dtype, py::name("to_numpy_dtype"), py::is_method(D));

    D.attr("convert_python_scalar") = py::cpp_function(&dtype::convert_python_scalar,
                                                       py::name("convert_python_scalar"),
                                                       py::is_method(D),
                                                       py::arg("value"));

    // Supports both Dtype.common(a, b) and a.common(b) / a.common(b, c, ...).
    D.attr("common") = py::cpp_function(
      [](Dtype first, const py::args& rest) {
          std::vector<Dtype> dtypes;
          dtypes.reserve(1 + rest.size());
          dtypes.push_back(first);
          for (py::handle h : rest)
              dtypes.push_back(h.cast<Dtype>());
          return dtype::common(dtypes);
      },
      py::name("common"),
      py::is_method(D));

    D.attr("from_numpy_dtype") = classmethod(
      py::cpp_function([](py::object /*cls*/,
                          py::object numpy_dtype) { return dtype::from_numpy_dtype(numpy_dtype); },
                       py::name("from_numpy_dtype"),
                       py::arg("cls"),
                       py::arg("dtype")));

    D.attr("__repr__") =
      py::cpp_function([](Dtype d) { return std::string("Dtype.") + dtype::repr(d); },
                       py::name("__repr__"),
                       py::is_method(D));

    D.attr("save_hdf5") = py::cpp_function(
      [](Dtype self, py::object hdf5_saver, py::object /*h5gr*/, const std::string& subpath) {
          using Underlying = std::underlying_type_t<Dtype>;
          hdf5_saver.attr("save")(static_cast<Underlying>(self), subpath + "value");
      },
      py::name("save_hdf5"),
      py::is_method(D),
      py::arg("hdf5_saver"),
      py::arg("h5gr"),
      py::arg("subpath"),
      "Export a Dtype enum member for cyten.tools.hdf5_io");

    D.attr("from_hdf5") = classmethod(py::cpp_function(
      [](py::object cls, py::object hdf5_loader, py::object h5gr, const std::string& subpath) {
          py::object value = hdf5_loader.attr("load")(subpath + "value");
          py::object obj = cls(value);
          hdf5_loader.attr("memorize_load")(h5gr, obj);
          return obj;
      },
      py::name("from_hdf5"),
      py::arg("cls"),
      py::arg("hdf5_loader"),
      py::arg("h5gr"),
      py::arg("subpath"),
      "Reconstruct a Dtype enum member from HDF5"));
}

} // namespace cyten
