#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/symmetries/symmetry.h>

#include <optional>
#include <string>

namespace cyten {

/// Get an instance of an appropriate tensor backend (cached).
///
/// Returns a Python object: C++ :class:`NoSymmetryBackend` when that path is selected,
/// otherwise the still-Python :class:`AbelianBackend` / :class:`FusionTreeBackend`.
///
/// Parameters mirror :func:`cyten.backends.backend_factory.get_backend`.
py::object get_backend(py::object symmetry = py::none(), py::object block_backend = py::none());

} // namespace cyten
