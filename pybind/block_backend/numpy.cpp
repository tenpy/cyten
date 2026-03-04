#include "../cyten_pybind11.h"
#include "trampolines.h"
#include <cyten/block_backend/numpy.h>

namespace cyten {

void
bind_block_backend_numpy(py::module_& m)
{

    py::class_<NumpyBlock, PyBlock<NumpyBlock>, py::smart_holder>(
      m, "NumpyBlock", "Block that holds a numpy array in a py::object.")
      .def(py::init<py::object>(), py::arg("arr"))
      .def("array", &NumpyBlock::array, py::return_value_policy::reference_internal)
      .def("__getitem__",
           [](NumpyBlock const& self, py::object key) -> py::object {
               py::object result = self.array().attr("__getitem__")(key);
               py::object sh = result.attr("shape");
               if (py::len(sh) == 0)
                   return result.attr("item")();
               return py::cast(std::make_shared<NumpyBlock>(result));
           })
      .def("__setitem__",
           [](NumpyBlock& self, py::object key, py::object value) {
               self.array().attr("__setitem__")(key, value);
           })
      .def(
        "__array__",
        [](NumpyBlock const& self, py::object dtype) {
            if (dtype.is_none())
                return self.array();
            return self.array().attr("astype")(dtype);
        },
        py::arg("dtype") = py::none())
      .def("__mul__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__mul__")(other));
           })
      .def("__rmul__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__rmul__")(other));
           })
      .def("__truediv__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__truediv__")(other));
           })
      .def("__add__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__add__")(other));
           })
      .def("__radd__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__radd__")(other));
           })
      .def("__sub__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__sub__")(other));
           })
      .def("__rsub__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__rsub__")(other));
           })
      .def_property_readonly("shape",
                             [](NumpyBlock const& self) { return self.array().attr("shape"); })
      .def_property_readonly("dtype",
                             [](NumpyBlock const& self) { return self.array().attr("dtype"); });

    py::class_<NumpyBlockBackend, PyBlockBackend<NumpyBlockBackend>, py::smart_holder>(
      m, "NumpyBlockBackend")
      .def(py::init<>());
}

} // namespace cyten
