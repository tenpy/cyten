#pragma once

#include <cyten/cyten.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cyten {

// the _core.cpp file implements the main pybind11 python module cyten._core with all python
// bindings.

// here, we have declarations of binding functions defined in the corresponding *.cpp files.

void bind_config(pybind11::module_& m);
void bind_version(pybind11::module_& m);

} // namespace cyten
