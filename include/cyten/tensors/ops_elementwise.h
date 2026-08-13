#pragma once

#include <cyten/cyten.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/tensors/forward_declare.h>

#include <optional>

namespace cyten {

/// The angle of a complex number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr angle(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar angle(BlockBackend::Scalar const& x);

/// The :ref:`elementwise <diagonal_elementwise>` cutoff inverse.
[[nodiscard]] DiagonalTensorPtr cutoff_inverse(DiagonalTensorCPtr x, float64 cutoff = 1e-15);
[[nodiscard]] BlockBackend::Scalar cutoff_inverse(BlockBackend::Scalar const& x,
                                                  float64 cutoff = 1e-15);

/// Complex conjugation, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr complex_conj(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar complex_conj(BlockBackend::Scalar const& x);

/// The imaginary part of a complex number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr imag(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar imag(BlockBackend::Scalar const& x);

/// The real part of a complex number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr real(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar real(BlockBackend::Scalar const& x);

/// If close to real, return the :func:`real` part, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr real_if_close(DiagonalTensorCPtr x, float64 tol = 100.);
[[nodiscard]] BlockBackend::Scalar real_if_close(BlockBackend::Scalar const& x, float64 tol = 100.);

/// The square root of a number, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr sqrt(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar sqrt(BlockBackend::Scalar const& x);

/// Stabilized logarithm, :ref:`elementwise <diagonal_elementwise>`.
[[nodiscard]] DiagonalTensorPtr stable_log(DiagonalTensorCPtr x, float64 cutoff = 1e-30);
[[nodiscard]] BlockBackend::Scalar stable_log(BlockBackend::Scalar const& x, float64 cutoff = 1e-30);

/// The exponential function.
///
/// For a tensor, viewed as a linear map from its domain to its codomain, the exponential
/// function is defined via its power series. For a diagonal tensor, this is equivalent to
/// the :ref:`elementwise <diagonal_elementwise>` exponential function.
[[nodiscard]] TensorPtr exp(TensorCPtr obj);

} // namespace cyten
