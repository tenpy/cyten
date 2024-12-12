#pragma once

// global definitions and includes for cyten

#include <vector>
#include <cstdint>
#include <string>
#include <array>
#include <stdexcept>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>


namespace cyten {

    namespace py = ::pybind11;
    typedef std::int64_t cyten_int;
} // namespace cyten
