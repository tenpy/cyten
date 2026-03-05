#include <cyten/block_backend/dtypes.h>
#include <cyten/block_backend/scalar.h>

namespace cyten {

float64
Scalar::real() const
{
    return value_.real();
}

float64
Scalar::cyten_double() const
{
    if (dtype_ == Dtype::Bool)
        throw std::runtime_error("Scalar::cyten_double: dtype is Bool");
    if (!dtype::is_real(dtype_))
        throw std::runtime_error("Scalar::cyten_double: dtype is not real (complex)");
    return value_.real();
}

complex128
Scalar::as_complex() const
{
    return value_;
}

bool
Scalar::as_bool() const
{
    if (dtype_ != Dtype::Bool)
        throw std::runtime_error("Scalar::as_bool: dtype is not Bool");
    return value_.real() != float64(0) || value_.imag() != float64(0);
}

py::object
Scalar::to_numpy() const
{
    py::object val;
    switch (dtype_) {
        case Dtype::Bool:
            val = py::cast(as_bool());
            break;
        case Dtype::Float32:
        case Dtype::Float64:
            val = py::cast(cyten_double());
            break;
        case Dtype::Complex64:
        case Dtype::Complex128:
            val = py::cast(as_complex());
            break;
    }
    return dtype::to_numpy_dtype(dtype_)(val);
}

} // namespace cyten
