#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/ops_elementwise.h>
#include <cyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include "docstrings/tensors/ops_elementwise.h"

#include <format>

namespace cyten {

namespace {

py::object
numpy()
{
    return py::module_::import("numpy");
}

bool
is_number_or_scalar(py::object x)
{
    return py::isinstance(x, py::module_::import("numbers").attr("Number")) ||
           py::isinstance(x, py::module_::import("cyten.block_backends").attr("Scalar")) ||
           (py::isinstance<Tensor>(x) &&
            py::module_::import("cyten.tensors._tensors").attr("is_scalar")(x).cast<bool>());
}

[[noreturn]] void
throw_elementwise_type_error(py::object x)
{
    throw py::type_error(std::format("Expected DiagonalTensor or scalar. Got {}",
                                     std::string(py::str(py::type::of(x)))));
}

template<typename TensorFn, typename ScalarPyFn>
py::object
dispatch_elementwise(py::object x, TensorFn tensor_fn, ScalarPyFn scalar_py_fn)
{
    if (py::isinstance<DiagonalTensor>(x)) {
        return py::cast(tensor_fn(x.cast<DiagonalTensorCPtr>()));
    }
    if (is_number_or_scalar(x)) {
        return scalar_py_fn(x);
    }
    throw_elementwise_type_error(x);
}

} // namespace

void
bind_tensors_ops_elementwise(py::module_& m)
{
    m.def(
      "angle",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return angle(t); },
            [](py::object n) { return numpy().attr("angle")(n); });
      },
      py::arg("x"),
      doc_plus(DOC(cyten, angle),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "cutoff_inverse",
      [](py::object x, float64 cutoff) {
          return dispatch_elementwise(
            x,
            [cutoff](DiagonalTensorCPtr t) { return cutoff_inverse(t, cutoff); },
            [cutoff](py::object n) {
                py::object abs_x = py::module_::import("builtins").attr("abs")(n);
                if (abs_x.cast<float64>() < cutoff) {
                    return py::object(py::int_(0));
                }
                return py::object(py::float_(1.0) / n);
            });
      },
      py::arg("x"),
      py::arg("cutoff") = 1e-15,
      doc_plus(DOC(cyten, cutoff_inverse),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "complex_conj",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return complex_conj(t); },
            [](py::object n) { return numpy().attr("conj")(n); });
      },
      py::arg("x"),
      doc_plus(DOC(cyten, complex_conj),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "imag",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return imag(t); },
            [](py::object n) { return numpy().attr("imag")(n); });
      },
      py::arg("x"),
      doc_plus(DOC(cyten, imag),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "real",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return real(t); },
            [](py::object n) { return numpy().attr("real")(n); });
      },
      py::arg("x"),
      doc_plus(DOC(cyten, real),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "real_if_close",
      [](py::object x, float64 tol) {
          return dispatch_elementwise(
            x,
            [tol](DiagonalTensorCPtr t) { return real_if_close(t, tol); },
            [tol](py::object n) {
                return numpy().attr("real_if_close")(n, py::arg("tol") = tol);
            });
      },
      py::arg("x"),
      py::arg("tol") = 100.,
      doc_plus(DOC(cyten, real_if_close),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "sqrt",
      [](py::object x) {
          return dispatch_elementwise(
            x,
            [](DiagonalTensorCPtr t) { return sqrt(t); },
            [](py::object n) { return numpy().attr("sqrt")(n); });
      },
      py::arg("x"),
      doc_plus(DOC(cyten, sqrt),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "stable_log",
      [](py::object x, float64 cutoff) {
          return dispatch_elementwise(
            x,
            [cutoff](DiagonalTensorCPtr t) { return stable_log(t, cutoff); },
            [cutoff](py::object n) {
                auto np = numpy();
                return np.attr("where")(np.attr("greater")(n, cutoff), np.attr("log")(n), 0.0);
            });
      },
      py::arg("x"),
      py::arg("cutoff") = 1e-30,
      doc_plus(DOC(cyten, stable_log),
               R"pydoc(
In Python, ``x`` may also be a number / :class:`~cyten.block_backends.Scalar`.
)pydoc"));

    m.def(
      "exp",
      [](py::object obj) {
          if (py::isinstance<Tensor>(obj)) {
              return py::cast(exp(obj.cast<TensorCPtr>()));
          }
          return numpy().attr("exp")(obj);
      },
      py::arg("obj"),
      doc_plus(DOC(cyten, exp),
               R"pydoc(
In Python, non-tensor inputs are forwarded to ``numpy.exp``.
)pydoc"));
}

} // namespace cyten
