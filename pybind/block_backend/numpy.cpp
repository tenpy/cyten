#include "trampolines.hpp"
#include <cyten/block_backend/numpy.h>

namespace cyten {

void
bind_block_backend_numpy(py::module_& m)
{

    py::class_<NumpyBlock, Block, PyBlock<NumpyBlock>, py::smart_holder>(
      m, "NumpyBlock", "Block that holds a numpy array in a py::object.")
      .def(py::init<py::array>(), py::arg("arr"))
      .def("to_numpy", &NumpyBlock::to_numpy, py::return_value_policy::reference_internal)
      .def("__getitem__",
           [](NumpyBlock const& self, py::object key) -> py::object {
               py::array result = self.to_numpy().attr("__getitem__")(key);
               py::object sh = result.attr("shape");
               if (py::len(sh) == 0)
                   return result.attr("item")();
               return py::cast(std::make_shared<NumpyBlock>(result));
           })
      .def("__setitem__",
           [](NumpyBlock& self, py::object key, py::object value) {
               self.to_numpy().attr("__setitem__")(key, value);
           })
      .def(
        "__array__",
        [](NumpyBlock const& self, py::object dtype) {
            if (dtype.is_none())
                return self.to_numpy();
            return py::cast<py::array>(self.to_numpy().attr("astype")(dtype));
        },
        py::arg("dtype") = py::none())
      .def("__mul__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.to_numpy().attr("__mul__")(other));
           })
      .def("__rmul__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.to_numpy().attr("__rmul__")(other));
           })
      .def("__truediv__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.to_numpy().attr("__truediv__")(other));
           })
      .def("__add__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.to_numpy().attr("__add__")(other));
           })
      .def("__radd__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.to_numpy().attr("__radd__")(other));
           })
      .def("__sub__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.to_numpy().attr("__sub__")(other));
           })
      .def("__rsub__", [](NumpyBlock const& self, py::object other) {
          return std::make_shared<NumpyBlock>(self.to_numpy().attr("__rsub__")(other));
      });

    py::
      class_<NumpyBlockBackend, BlockBackend, PyBlockBackend<NumpyBlockBackend>, py::smart_holder>(
        m, "NumpyBlockBackend")
        .def(py::init<>());
}

} // namespace cyten
