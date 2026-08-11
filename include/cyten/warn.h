#pragma once

#include <pybind11/warnings.h>

#include <Python.h>
#include <iostream>
#include <string>
#include <string_view>

namespace cyten {

/// Emit a UserWarning via Python's warning machinery when the interpreter is active.
/// Falls back to stderr when ``!Py_IsInitialized()`` (pure-C++ / pre-interpreter use).
inline void
warn(std::string_view message, int stack_level = 2)
{
    if (!Py_IsInitialized()) {
        // Fallback for pure-C++ / pre-interpreter use; refine later if needed.
        std::cerr << "UserWarning: " << message << '\n';
        return;
    }
    pybind11::warnings::warn(std::string(message).c_str(), PyExc_UserWarning, stack_level);
}

} // namespace cyten
