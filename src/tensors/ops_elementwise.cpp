#include <cyten/tensors/ops_elementwise.h>

#include <cyten/tensors/diagonal_tensor.h>

#include <format>
#include <stdexcept>

namespace cyten {

namespace {

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

py::object
numpy()
{
    return py::module_::import("numpy");
}

bool
is_diagonal_tensor(py::object x)
{
    return py::isinstance(x, tensors_mod().attr("DiagonalTensor"))
           || py::isinstance<DiagonalTensor>(x);
}

bool
is_scalar_obj(py::object x)
{
    return tensors_mod().attr("is_scalar")(x).cast<bool>();
}

[[noreturn]] void
throw_elementwise_type_error(py::object x)
{
    throw py::type_error(
      std::format("Expected DiagonalTensor or scalar. Got {}",
                  std::string(py::str(py::type::of(x)))));
}

/// DiagonalTensor path: ``block_backend.<name>(block, **kwargs)`` via ``_elementwise_unary``.
py::object
elementwise_on_diagonal(py::object x,
                        char const* block_func,
                        bool maps_zero_to_zero,
                        py::dict kwargs)
{
    py::object meth = x.attr("backend").attr("block_backend").attr(block_func);
    py::cpp_function unary([meth, kwargs](py::object block) {
        return meth(block, **kwargs);
    });
    return x.attr("_elementwise_unary")(unary, py::arg("maps_zero_to_zero") = maps_zero_to_zero);
}

} // namespace

py::object
angle(py::object x)
{
    if (is_diagonal_tensor(x)) {
        return elementwise_on_diagonal(x, "angle", true, py::dict());
    }
    if (is_scalar_obj(x)) {
        return numpy().attr("angle")(x);
    }
    throw_elementwise_type_error(x);
}

py::object
cutoff_inverse(py::object x, float64 cutoff)
{
    if (is_diagonal_tensor(x)) {
        py::dict kw;
        kw["cutoff"] = cutoff;
        return elementwise_on_diagonal(x, "cutoff_inverse", true, kw);
    }
    if (is_scalar_obj(x)) {
        // The cutoff-inverse for a number ``x`` is ``1 / x`` if ``abs(x) >= cutoff``, otherwise ``0``.
        py::object abs_x = py::module_::import("builtins").attr("abs")(x);
        if (abs_x.cast<float64>() < cutoff) {
            return py::int_(0);
        }
        return py::float_(1.0) / x;
    }
    throw_elementwise_type_error(x);
}

py::object
complex_conj(py::object x)
{
    if (is_diagonal_tensor(x)) {
        return elementwise_on_diagonal(x, "conj", true, py::dict());
    }
    if (is_scalar_obj(x)) {
        return numpy().attr("conj")(x);
    }
    throw_elementwise_type_error(x);
}

py::object
imag(py::object x)
{
    if (is_diagonal_tensor(x)) {
        return elementwise_on_diagonal(x, "imag", true, py::dict());
    }
    if (is_scalar_obj(x)) {
        return numpy().attr("imag")(x);
    }
    throw_elementwise_type_error(x);
}

py::object
real(py::object x)
{
    if (is_diagonal_tensor(x)) {
        return elementwise_on_diagonal(x, "real", true, py::dict());
    }
    if (is_scalar_obj(x)) {
        return numpy().attr("real")(x);
    }
    throw_elementwise_type_error(x);
}

py::object
real_if_close(py::object x, float64 tol)
{
    if (is_diagonal_tensor(x)) {
        py::dict kw;
        kw["tol"] = tol;
        return elementwise_on_diagonal(x, "real_if_close", true, kw);
    }
    if (is_scalar_obj(x)) {
        return numpy().attr("real_if_close")(x, py::arg("tol") = tol);
    }
    throw_elementwise_type_error(x);
}

py::object
sqrt(py::object x)
{
    if (is_diagonal_tensor(x)) {
        return elementwise_on_diagonal(x, "sqrt", true, py::dict());
    }
    if (is_scalar_obj(x)) {
        return numpy().attr("sqrt")(x);
    }
    throw_elementwise_type_error(x);
}

py::object
stable_log(py::object x, float64 cutoff)
{
    if (!(cutoff > 0)) {
        throw std::runtime_error("cutoff must be > 0");
    }
    if (is_diagonal_tensor(x)) {
        py::dict kw;
        kw["cutoff"] = cutoff;
        return elementwise_on_diagonal(x, "stable_log", true, kw);
    }
    if (is_scalar_obj(x)) {
        auto np = numpy();
        return np.attr("where")(np.attr("greater")(x, cutoff), np.attr("log")(x), 0.0);
    }
    throw_elementwise_type_error(x);
}

} // namespace cyten
