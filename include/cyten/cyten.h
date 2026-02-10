#pragma once

// global definitions and includes for cyten

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <stdfloat>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static_assert(std::numeric_limits<double>::is_iec559);

namespace cyten {
namespace py = ::pybind11;
using cyten_int = std::int64_t;
using cyten_float = double;
using cyten_complex = std::complex<cyten_float>;
using size_t = std::size_t;
} // namespace cyten
