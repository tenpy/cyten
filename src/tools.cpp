#include <cyten/tools.h>

#include <pybind11/numpy.h>

namespace cyten {

// CHECKME: the following was appended by .cursor/skills/pybind11-codegen/pybind11_codegen.py
// gen_cpp_definition --py-name format_like_list --header-file include/cyten/tools.h --src-file
// src/tools.cpp
/// Format elements of an iterable as if it were a plain list.
std::string
format_like_list(py::iterable it)
{
    std::string result = "[";
    bool first = true;
    for (auto&& item : it) {
        if (!first)
            result += ", ";
        result += py::str(item).cast<std::string>();
        first = false;
    }
    result += "]";
    return result;
}

bool
is_iterable(py::object a)
{
    try {
        py::iter(a);
        return true;
    } catch (py::error_already_set& m) {
        // expected error: TypeError if not iterable
        if (!m.matches(PyExc_TypeError))
            throw;
    }
    return false;
}

py::object
to_iterable(py::object a)
{
    if (!py::isinstance<py::str>(a) && is_iterable(a))
        return a;
    py::list result(1);
    result[0] = a;
    return result;
}

cyten_int
to_valid_idx(cyten_int idx, cyten_int length)
{
    if (idx < -length || idx >= length)
        throw std::out_of_range("Index " + std::to_string(idx) + " out of bounds for length " +
                                std::to_string(length));
    if (idx < 0)
        idx += length;
    return idx;
}

} // namespace cyten
