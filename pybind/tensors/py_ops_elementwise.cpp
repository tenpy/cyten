#include <cyten/tensors/ops_elementwise.h>

#include "../py_cyten_pybind11.h"

namespace cyten {

void
bind_tensors_ops_elementwise(py::module_& m)
{
    m.def("angle",
          &angle,
          py::arg("x"),
          R"pydoc(
The angle of a complex number, :ref:`elementwise <diagonal_elementwise>`.

The counterclockwise angle from the positive real axis on the complex plane in the
range (-pi, pi] with a real dtype. The angle of `0.` is `0.`.
)pydoc");

    m.def("cutoff_inverse",
          &cutoff_inverse,
          py::arg("x"),
          py::arg("cutoff") = 1e-15,
          R"pydoc(
The :ref:`elementwise <diagonal_elementwise>` cutoff inverse.

The cutoff-inverse for a number ``x`` is ``1 / x`` if ``abs(x) >= cutoff``, otherwise ``0``.
)pydoc");

    m.def("complex_conj",
          &complex_conj,
          py::arg("x"),
          R"pydoc(Complex conjugation, :ref:`elementwise <diagonal_elementwise>`.)pydoc");

    m.def("imag",
          &imag,
          py::arg("x"),
          R"pydoc(The imaginary part of a complex number, :ref:`elementwise <diagonal_elementwise>`.)pydoc");

    m.def("real",
          &real,
          py::arg("x"),
          R"pydoc(The real part of a complex number, :ref:`elementwise <diagonal_elementwise>`.)pydoc");

    m.def("real_if_close",
          &real_if_close,
          py::arg("x"),
          py::arg("tol") = 100.,
          R"pydoc(
If close to real, return the :func:`real` part, :ref:`elementwise <diagonal_elementwise>`.

Parameters
----------
x : :class:`DiagonalTensor` | Number
    The input complex number(s)
tol : float
    The precision for considering the imaginary part "close to zero".
    Multiples of machine epsilon for the dtype of `x`.

Returns
-------
If `x` is close to real, the real part of `x`. Otherwise the original complex `x`.
)pydoc");

    m.def("sqrt",
          &sqrt,
          py::arg("x"),
          R"pydoc(The square root of a number, :ref:`elementwise <diagonal_elementwise>`.)pydoc");

    m.def("stable_log",
          &stable_log,
          py::arg("x"),
          py::arg("cutoff") = 1e-30,
          R"pydoc(
Stabilized logarithm, :ref:`elementwise <diagonal_elementwise>`.

For values ``> cutoff``, this is the standard natural logarithm. For values smaller than the
cutoff, return 0.
)pydoc");
}

} // namespace cyten
