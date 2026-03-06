// NOTE: this file is #included from py_block_backend.cpp

#include "py_trampolines.hpp"
#include <cyten/block_backend/numpy.h>

namespace cyten {

void
bind_block_backend_numpy(py::module_& m)
{

    py::
      class_<NumpyBlockBackend, BlockBackend, PyBlockBackend<NumpyBlockBackend>, py::smart_holder>
        numpy_block_backend(m, "NumpyBlockBackend");
    numpy_block_backend.doc() = "A block backend using numpy.";
    numpy_block_backend.def(py::init<>());

    py::class_<NumpyBlockBackend::Block,
               BlockBackend::Block,
               PyBlock<NumpyBlockBackend::Block>,
               py::smart_holder>(
      numpy_block_backend, "BlockCls", "Block that holds a numpy array in a py::object.")
      .def(py::init<py::array>(), py::arg("arr"))
      .def("to_numpy",
           &NumpyBlockBackend::Block::to_numpy,
           py::return_value_policy::reference_internal)
      .def("__getitem__",
           [](const NumpyBlockBackend::Block& self, py::object key) -> py::object {
               py::array result = self.to_numpy().attr("__getitem__")(key);
               py::object sh = result.attr("shape");
               if (py::len(sh) == 0)
                   return result.attr("item")();
               return py::cast(std::make_shared<NumpyBlockBackend::Block>(result));
           })
      .def("__setitem__",
           [](NumpyBlockBackend::Block& self, py::object key, py::object value) {
               self.to_numpy().attr("__setitem__")(key, value);
           })
      .def(
        "__array__",
        [](const NumpyBlockBackend::Block& self, py::object dtype) {
            if (dtype.is_none())
                return self.to_numpy();
            return py::cast<py::array>(self.to_numpy().attr("astype")(dtype));
        },
        py::arg("dtype") = py::none())
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
