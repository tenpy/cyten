#include <lasyten/tools.h>

#include "doc_plus.h"
#include "docstrings/tools.h"
#include "py_lasyten_pybind11.h"

namespace py = pybind11;
namespace lasyten {

void
bind_tools(py::module_& m)
{
    // Translate C++ NotImplemented into Python's builtin NotImplementedError.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (NotImplemented const& e) {
            PyErr_SetString(PyExc_NotImplementedError, e.what());
        }
    });

    m.def("format_like_list",
          &lasyten::format_like_list,
          doc_cpp_ref(R"pydoc(format_like_list)pydoc", "lasyten::format_like_list"),
          py::arg("it"));

    m.def("is_iterable", &lasyten::is_iterable, py::arg("a"), "If the given object is iterable.");

    m.def("to_iterable",
          &lasyten::to_iterable,
          py::arg("a"),
          "If `a` is a not iterable or a string, return [a], else return a.");

    m.def("to_valid_idx",
          &lasyten::to_valid_idx,
          py::arg("idx"),
          py::arg("length"),
          "Convert to a valid non-negative index into the given length.");

    m.def("permutation_as_swaps",
          &lasyten::permutation_as_swaps,
          py::arg("permutation"),
          DOC(lasyten, permutation_as_swaps));
}

} // namespace lasyten
