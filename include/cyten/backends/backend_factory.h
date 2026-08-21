#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/symmetries/symmetry.h>

#include <optional>
#include <string>

namespace cyten {

/// Get an instance of an appropriate tensor backend (cached).
///
/// Returns a Python object wrapping a C++ `NoSymmetryBackend`,
/// `AbelianBackend`, or `FusionTreeBackend`.
///
/// Parameters mirror `get_backend`.
/// Get an instance of an appropriate backend.
///
/// Backends are instantiated only once and then cached. If a suitable backend instance is in
/// the cache, that same instance is returned.
///
/// @param symmetry Specifies which subclass of `TensorBackend` to use, either directly via string,
/// or as the minimal version which supports the given symmetry.
/// @param block_backend Specify which block backend to use.
py::object get_backend(py::object symmetry = py::none(), py::object block_backend = py::none());

/// Typed overload: pick a backend that supports `symmetry`.
TensorBackend::Ptr get_backend(Symmetry::Ptr symmetry,
                               std::optional<std::string> block_backend = std::nullopt);

} // namespace cyten
