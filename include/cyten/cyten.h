#pragma once

// global definitions and includes for cyten

#include <vector>
#include <cstdint>
#include <stdfloat>
#include <string>
#include <array>
#include <stdexcept>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#if __STDCPP_FLOAT64_T__ != 1
    #error "64-bit float type required"
#endif

namespace cyten {
    namespace py = ::pybind11;
    typedef std::int64_t cyten_int;
    typedef std::float64_t cyten_float;
    typedef std::complex<cyten_float> cyten_complex;

} // namespace cyten
