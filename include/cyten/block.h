#pragma once

#include <cyten/cyten.h>
#include <cyten/dtypes.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyten {

/// Holds a single scalar value with a Dtype. Use accessors to cast to the desired C++ type.
class Scalar
{
  public:
    Scalar(Dtype dtype, cyten_complex value)
      : dtype_(dtype)
      , value_(value)
    {
    }

    /// Construct from bool; dtype is Bool.
    explicit Scalar(bool b)
      : dtype_(Dtype::Bool)
      , value_(b ? cyten_float(1) : cyten_float(0))
    {
    }

    /// Construct from real; dtype is Float64.
    explicit Scalar(cyten_float x)
      : dtype_(Dtype::Float64)
      , value_(x)
    {
    }

    /// Construct from complex; dtype is Complex128.
    explicit Scalar(cyten_complex z)
      : dtype_(Dtype::Complex128)
      , value_(z)
    {
    }

    Dtype dtype() const { return dtype_; }

    /// Real part; valid for any dtype (complex -> real part, bool -> 0 or 1).
    cyten_float real() const;

    /// As a real (float) scalar. Throws if dtype is not Float32 or Float64.
    cyten_float cyten_double() const;

    /// As a complex scalar. Always valid (real/bool stored with zero imaginary part).
    cyten_complex as_complex() const;

    /// As a bool. Throws if dtype is not Bool.
    bool as_bool() const;

    /// Return as a numpy scalar (np.bool_, np.float32, np.float64, np.complex64, np.complex128).
    py::object to_numpy() const;

  private:
    Dtype dtype_;
    cyten_complex value_;
};

/// Abstract base class for dense blocks. Subclassed per backend (e.g. NumpyBlock).
/// Pass and return as std::shared_ptr<Block>; use const std::shared_ptr<Block>& for inputs.
class Block
{
  public:
    virtual ~Block() = default;

    /// Shape of the block (one size per axis).
    virtual std::vector<cyten_int> shape() const = 0;

    /// Dtype of the block entries.
    virtual Dtype dtype() const = 0;

    /// Device string (e.g. "cpu", "cuda:0").
    virtual std::string device() const = 0;

    /// Element or sub-block access by indices. Returns a scalar (py::object) or a new Block for
    /// slices.
    virtual py::object operator[](std::vector<cyten_int> const& idcs) const = 0;
};

using BlockPtr = std::shared_ptr<Block>;
using BlockCPtr = std::shared_ptr<const Block>;

} // namespace cyten
