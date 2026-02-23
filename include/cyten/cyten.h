#pragma once

// global definitions and includes for cyten

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <stdfloat>
#include <string>

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

class NotImplemented : public std::logic_error
{
  public:
    NotImplemented(std::string name);
    // TODO JH: definition for now in check.cpp, move to tools or something like that.
};

} // namespace cyten
