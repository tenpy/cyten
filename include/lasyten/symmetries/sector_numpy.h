#pragma once

/// Sector / SectorArray ↔ NumPy helpers for liblasyten (no type_casters required).

#include "../lasyten.h"
#include "sector.h"

namespace lasyten {

py::array sector_to_numpy(Sector const& src);
py::array sector_array_to_numpy(SectorArray const& src);

Sector sector_from_numpy(py::handle src);
SectorArray sector_array_from_numpy(py::handle src);

} // namespace lasyten
