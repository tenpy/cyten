#pragma once

/// Sector / SectorArray ↔ NumPy without relying on pybind type_casters.
///
/// Use these from ``src/`` (libcyten). Binding TUs still include
/// ``pybind/symmetries/casters.hpp`` for automatic ``py::cast``.

#include "../cyten.h"
#include "sector.h"

namespace cyten {

py::array sector_to_numpy(Sector const& src);
py::array sector_array_to_numpy(SectorArray const& src);

Sector sector_from_numpy(py::handle src);
SectorArray sector_array_from_numpy(py::handle src);

} // namespace cyten
