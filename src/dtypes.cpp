#include <cyten/dtypes.h>

namespace cyten {

namespace dtype {

py::module_&
numpy_module()
{
    static py::module_ np = py::module_::import("numpy");
    return np;
}

std::string
repr(Dtype dtype)
{
    switch (dtype) {
        case Dtype::Bool:
            return "bool";
        case Dtype::Float32:
            return "float32";
        case Dtype::Complex64:
            return "complex64";
        case Dtype::Float64:
            return "float64";
        case Dtype::Complex128:
            return "complex128";
        default:
            return "?";
    }
}

Dtype
to_complex(Dtype dtype)
{
    if (dtype == Dtype::Bool)
        throw std::invalid_argument("Dtype.bool can not be converted to complex");
    if (static_cast<std::uint8_t>(dtype) % 2 == 1)
        return dtype;
    return static_cast<Dtype>(static_cast<std::uint8_t>(dtype) + 1);
}

Dtype
to_real(Dtype dtype)
{
    if (dtype == Dtype::Bool)
        throw std::invalid_argument("Dtype.bool can not be converted to real");
    if (static_cast<std::uint8_t>(dtype) % 2 == 0)
        return dtype;
    return static_cast<Dtype>(static_cast<std::uint8_t>(dtype) - 1);
}

double
eps(Dtype dtype)
{
    if (dtype == Dtype::Bool)
        throw std::invalid_argument("Dtype.bool is not inexact");
    std::uint8_t n_bits = 8 * (static_cast<std::uint8_t>(dtype) / 2);
    if (n_bits == 32)
        return std::pow(2.0, -23); // float32
    if (n_bits == 64)
        return std::pow(2.0, -52); // float64
    throw NotImplemented(std::string("Dtype.eps not implemented for n_bits=") +
                         std::to_string(n_bits));
}

py::object
to_numpy_dtype(Dtype dtype)
{
    py::module_ np = numpy_module();
    switch (dtype) {
        case Dtype::Bool:
            return np.attr("bool_");
        case Dtype::Float32:
            return np.attr("float32");
        case Dtype::Float64:
            return np.attr("float64");
        case Dtype::Complex64:
            return np.attr("complex64");
        case Dtype::Complex128:
            return np.attr("complex128");
        default:
            throw std::invalid_argument("unknown Dtype");
    }
}

Dtype
from_numpy_dtype(py::object numpy_dtype)
{
    if (numpy_dtype.is_none())
        throw std::invalid_argument("None is not a valid cyten dtype");
    py::module_ np = numpy_module();
    if (numpy_dtype.equal(np.attr("bool_")))
        return Dtype::Bool;
    if (numpy_dtype.equal(np.attr("float32")))
        return Dtype::Float32;
    if (numpy_dtype.equal(np.attr("float64")))
        return Dtype::Float64;
    if (numpy_dtype.equal(np.attr("complex64")))
        return Dtype::Complex64;
    if (numpy_dtype.equal(np.attr("complex128")))
        return Dtype::Complex128;
    throw std::invalid_argument("unknown numpy dtype");
}

Dtype
common(const std::vector<Dtype>& dtypes)
{
    if (dtypes.empty())
        throw std::invalid_argument("common_dtype requires at least one dtype");
    Dtype res = dtypes.front();
    for (const auto& t : dtypes)
        if (static_cast<std::uint8_t>(t) > static_cast<std::uint8_t>(res))
            res = t;
    if (is_real(res)) {
        for (Dtype t : dtypes) {
            if (is_complex(t)) {
                res = to_complex(res);
                break;
            }
        }
    }
    return res;
}

py::object
convert_python_scalar(Dtype dtype, py::object value)
{
    if (dtype == Dtype::Bool) {
        // bool: accept True, False, 0, 1
        if (value.equal(py::cast(true)) || value.equal(py::cast(false)) ||
            value.equal(py::cast(0)) || value.equal(py::cast(1)))
            return py::cast(bool(py::cast<int>(value)));
        throw std::invalid_argument("Type incompatible with dtype bool");
    }
    if (is_real(dtype)) {
        try {
            return py::cast(py::cast<double>(value));
        } catch (const py::cast_error&) {
            try {
                return py::cast(static_cast<double>(py::cast<int>(value)));
            } catch (const py::cast_error&) {
                throw std::invalid_argument("Type incompatible with real dtype");
            }
        }
    }
    // complex
    try {
        return py::cast(py::cast<cyten_complex>(value));
    } catch (const py::cast_error&) {
        try {
            return py::cast(cyten_complex(py::cast<double>(value), 0));
        } catch (const py::cast_error&) {
            throw std::invalid_argument("Type incompatible with complex dtype");
        }
    }
}

py::object
python_type(Dtype dtype)
{
    if (dtype == Dtype::Bool)
        return py::module_::import("builtins").attr("bool");
    if (is_real(dtype))
        return py::module_::import("builtins").attr("float");
    return py::module_::import("builtins").attr("complex");
}

py::object
zero_scalar(Dtype dtype)
{
    if (dtype == Dtype::Bool)
        return py::cast(false);
    if (is_real(dtype))
        return py::cast(0.0);
    return py::cast(cyten_complex(0.0, 0.0));
}

} // namespace dtype

} // namespace cyten
