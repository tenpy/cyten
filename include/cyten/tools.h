#pragma once

#include <cyten/cyten.h>
#include <string>

namespace cyten {

/// Format elements of an iterable as if it were a plain list.
std::string format_like_list(py::iterable it);

/// If the given object is iterable.
bool is_iterable(py::object a);

/// If `a` is not iterable or a string, return [a], else return a.
py::object to_iterable(py::object a);

/// Convert to a valid non-negative index into the given length.
cyten_int to_valid_idx(cyten_int idx, cyten_int length);

} // namespace cyten
