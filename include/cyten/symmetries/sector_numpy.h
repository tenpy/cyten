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

/// Wrap a Sector as the bound ``cyten._core.Sector`` so ``Hdf5Saver.save`` calls ``save_hdf5``.
py::object sector_as_hdf5_exportable(Sector const& src);
py::object sector_array_as_hdf5_exportable(SectorArray const& src);

/// Load a Sector saved either as bound Sector (``save_hdf5``) or as a plain ndarray.
Sector sector_from_hdf5_object(py::handle src);
SectorArray sector_array_from_hdf5_object(py::handle src);

} // namespace cyten
