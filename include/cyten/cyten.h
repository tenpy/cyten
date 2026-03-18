#pragma once

// global definitions and includes for cyten

#include <cassert>
#include <cstdint>
#include <stdfloat>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static_assert(std::numeric_limits<double>::is_iec559); // double is indeed 64 bit

namespace cyten {

namespace py = ::pybind11;
using int64 = std::int64_t;
using float32 = float;
using float64 = double;
using complex64 = std::complex<float32>;
using complex128 = std::complex<float64>;

} // namespace cyten
