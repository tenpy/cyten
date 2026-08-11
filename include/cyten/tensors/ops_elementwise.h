#pragma once

#include <cyten/cyten.h>

#include <optional>

namespace cyten {

/// The angle of a complex number, :ref:`elementwise <diagonal_elementwise>`.
///
/// The counterclockwise angle from the positive real axis on the complex plane in the
/// range (-pi, pi] with a real dtype. The angle of `0.` is `0.`.
[[nodiscard]] py::object angle(py::object x);

/// The :ref:`elementwise <diagonal_elementwise>` cutoff inverse.
///
/// The cutoff-inverse for a number ``x`` is ``1 / x`` if ``abs(x) >= cutoff``, otherwise ``0``.
[[nodiscard]] py::object cutoff_inverse(py::object x, float64 cutoff = 1e-15);

/// Complex conjugation, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] py::object complex_conj(py::object x);

/// The imaginary part of a complex number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] py::object imag(py::object x);

/// The real part of a complex number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] py::object real(py::object x);

/// If close to real, return the :func:`real` part, :ref:`elementwise <diagonal_elementwise>`.
///
/// Parameters
/// ----------
/// x : :class:`DiagonalTensor` | Number
///     The input complex number(s)
/// tol : float
///     The precision for considering the imaginary part "close to zero".
///     Multiples of machine epsilon for the dtype of `x`.
///
/// Returns
/// -------
/// If `x` is close to real, the real part of `x`. Otherwise the original complex `x`.
[[nodiscard]] py::object real_if_close(py::object x, float64 tol = 100.);

/// The square root of a number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] py::object sqrt(py::object x);

/// Stabilized logarithm, :ref:`elementwise <diagonal_elementwise>`.
///
/// For values ``> cutoff``, this is the standard natural logarithm. For values smaller than the
/// cutoff, return 0.
[[nodiscard]] py::object stable_log(py::object x, float64 cutoff = 1e-30);

} // namespace cyten
