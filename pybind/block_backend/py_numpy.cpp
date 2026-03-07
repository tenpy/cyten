// NOTE: this file is #included from py_block_backend.cpp

#include "py_trampolines.hpp"
#include <cyten/block_backend/numpy.h>

namespace cyten {

void
bind_block_backend_numpy(py::module_& m)
{

    py::class_<NumpyBlockBackend, BlockBackend, py::smart_holder> numpy_block_backend(
      m, "NumpyBlockBackend");
    numpy_block_backend.doc() = R"pydoc(
        A block backend using numpy.

        No constructor available, use from_factory instead.
        Not to be subclassed.
        )pydoc";
    numpy_block_backend.def_static(
      "from_factory",
      &NumpyBlockBackend::from_factory,
      py::arg("device") = "cpu",
      py::return_value_policy::reference,
      "Get the backend instance for the given device (nearly-singleton per device).");
    numpy_block_backend.def_static("load_hdf5",
                                   &NumpyBlockBackend::load_hdf5,
                                   py::arg("hdf5_loader"),
                                   py::arg("h5gr"),
                                   py::arg("subpath"),
                                   "Load a block from an HDF5 file.");

    py::class_<NumpyBlockBackend::Block, BlockBackend::Block, py::smart_holder>(
      numpy_block_backend, "BlockCls", "Block that holds a numpy array in a py::object.")
      .def(py::init<py::array>(), py::arg("arr"))
      .def("to_numpy",
           py::overload_cast<>(&NumpyBlockBackend::Block::to_numpy, py::const_),
           py::return_value_policy::reference_internal)
      .def("to_numpy",
           py::overload_cast<Dtype>(&NumpyBlockBackend::Block::to_numpy, py::const_),
           py::arg("dtype"),
           py::return_value_policy::reference_internal)
      //  .def(
      //    "__array__",
      //    [](const NumpyBlockBackend::Block& self, py::object dtype) {
      //        if (dtype.is_none())
      //            return self.to_numpy();
      //        return py::cast<py::array>(self.to_numpy().attr("astype")(dtype));
      //    },
      //    py::arg("dtype") = py::none())
      .def("__mul__",
           [](const NumpyBlockBackend::Block& self, py::object other) {
               return std::make_shared<NumpyBlockBackend::Block>(
                 self.to_numpy().attr("__mul__")(other));
           })
      .def("__rmul__",
           [](const NumpyBlockBackend::Block& self, py::object other) {
               return std::make_shared<NumpyBlockBackend::Block>(
                 self.to_numpy().attr("__rmul__")(other));
           })
      .def("__truediv__",
           [](const NumpyBlockBackend::Block& self, py::object other) {
               return std::make_shared<NumpyBlockBackend::Block>(
                 self.to_numpy().attr("__truediv__")(other));
           })
      .def("__add__",
           [](const NumpyBlockBackend::Block& self, py::object other) {
               return std::make_shared<NumpyBlockBackend::Block>(
                 self.to_numpy().attr("__add__")(other));
           })
      .def("__radd__",
           [](const NumpyBlockBackend::Block& self, py::object other) {
               return std::make_shared<NumpyBlockBackend::Block>(
                 self.to_numpy().attr("__radd__")(other));
           })
      .def("__sub__",
           [](const NumpyBlockBackend::Block& self, py::object other) {
               return std::make_shared<NumpyBlockBackend::Block>(
                 self.to_numpy().attr("__sub__")(other));
           })
      .def("__rsub__", [](const NumpyBlockBackend::Block& self, py::object other) {
          return std::make_shared<NumpyBlockBackend::Block>(
            self.to_numpy().attr("__rsub__")(other));
      });
}

} // namespace cyten
