#include <cyten/tools.h>

#include "cyten_pybind11.h"

namespace py = pybind11;
namespace cyten {

void
bind_tools(py::module_& m)
{

    m.def("format_like_list",
          &cyten::format_like_list,
          R"pydoc(
          Format elements of an iterable as if it were a plain list.

          This means surrounding them with brackets and separating them by `', '`.
          )pydoc",
          py::arg("it"));

    m.def("is_iterable", &cyten::is_iterable, py::arg("a"), "If the given object is iterable.");

    m.def("to_iterable",
          &cyten::to_iterable,
          py::arg("a"),
          "If `a` is a not iterable or a string, return [a], else return a.");

    m.def("to_valid_idx",
          &cyten::to_valid_idx,
          py::arg("idx"),
          py::arg("length"),
          "Convert to a valid non-negative index into the given length.");
}

} // namespace cyten
