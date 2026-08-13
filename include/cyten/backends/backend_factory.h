#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/symmetries/symmetry.h>

#include <optional>
#include <string>

namespace cyten {

/// Get an instance of an appropriate tensor backend (cached).
///
/// Returns a Python object wrapping a C++ :class:`NoSymmetryBackend`,
/// :class:`AbelianBackend`, or :class:`FusionTreeBackend`.
///
/// Parameters mirror :func:`cyten.backends.backend_factory.get_backend`.
py::object get_backend(py::object symmetry = py::none(), py::object block_backend = py::none());

/// Typed overload: pick a backend that supports `symmetry`.
TensorBackend::Ptr get_backend(Symmetry::Ptr symmetry,
                               std::optional<std::string> block_backend = std::nullopt);

} // namespace cyten
