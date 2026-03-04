#pragma once

#include <cstdint>
#include <cyten/cyten.h>
#include <string>
#include <vector>

namespace cyten {

/// The dtype of (entries in) a tensor.
enum class Dtype : std::uint8_t
{
    // value = num_bytes * 2 + int(not is_real)
    Bool = 2,
    Float32 = 8,
    Complex64 = 9,
    Float64 = 16,
    Complex128 = 17,
};

namespace dtype {

constexpr bool
is_real(Dtype dtype)
{
    return static_cast<std::uint8_t>(dtype) % 2 == 0;
}
constexpr bool
is_complex(Dtype dtype)
{
    return static_cast<std::uint8_t>(dtype) % 2 == 1;
}

Dtype to_complex(Dtype dtype);
Dtype to_real(Dtype dtype);

std::string repr(Dtype dtype);

/// Epsilon: difference between 1.0 and next representable. Bool raises.
double eps(Dtype dtype);

/// Numpy dtype object for this Dtype.
py::object to_numpy_dtype(Dtype dtype);

/// Build Dtype from numpy dtype object. Returns none for None input (Python only).
Dtype from_numpy_dtype(py::object numpy_dtype);

/// Common dtype that can represent all given dtypes.
Dtype common(const std::vector<Dtype>& dtypes);

/// Convert a Python scalar to this dtype's scalar type (returns py::object).
py::object convert_python_scalar(Dtype dtype, py::object value);

/// Python type (bool, float, complex) as a type object.
py::object python_type(Dtype dtype);

/// Zero scalar in this dtype (e.g. 0, 0.0, 0j).
py::object zero_scalar(Dtype dtype);

} // namespace dtype

} // namespace cyten
