#pragma once

#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>

namespace cyten {

/// Holds a single scalar value with a Dtype. Use accessors to cast to the desired C++ type.
class Scalar
{
  public:
    Scalar(Dtype dtype, complex128 value)
      : dtype_(dtype)
      , value_(value)
    {
    }

    /// Construct from bool; dtype is Bool.
    Scalar(bool b)
      : dtype_(Dtype::Bool)
      , value_(b ? float64(1) : float64(0))
    {
    }

    /// Construct from real; dtype is Float64.
    Scalar(float64 x)
      : dtype_(Dtype::Float64)
      , value_(x)
    {
    }

    /// Construct from complex; dtype is Complex128.
    Scalar(complex128 z)
      : dtype_(Dtype::Complex128)
      , value_(z)
    {
    }

    Dtype dtype() const { return dtype_; }

    /// Real part; valid for any dtype (complex -> real part, bool -> 0 or 1).
    float64 real() const;

    /// As a real (float) scalar. Throws if dtype is not Float32 or Float64.
    float64 cyten_double() const;

    /// As a complex scalar. Always valid (real/bool stored with zero imaginary part).
    complex128 as_complex() const;

    /// As a bool. Throws if dtype is not Bool.
    bool as_bool() const;

    /// Return as a numpy scalar (np.bool_, np.float32, np.float64, np.complex64, np.complex128).
    py::object to_numpy() const;

  private:
    Dtype dtype_;
    complex128 value_;
};

} // namespace cyten
