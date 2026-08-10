#pragma once

/// Sector / SectorArray ↔ NumPy helpers for libcyten (no type_casters required).

#include "../cyten.h"
#include "sector.h"

namespace cyten {

py::array sector_to_numpy(Sector const& src);
py::array sector_array_to_numpy(SectorArray const& src);

Sector sector_from_numpy(py::handle src);
SectorArray sector_array_from_numpy(py::handle src);

} // namespace cyten
