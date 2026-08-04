#pragma once

/// HDF5-exportable wrappers for Sector / SectorArray.
///
/// Distinct from ``Sector`` / ``SectorArray`` so pybind can bind these as Python
/// ``Sector`` / ``SectorArray`` with ``save_hdf5`` / ``from_hdf5`` without
/// conflicting with the ndarray type casters used for the rest of the API.

#include "sector.h"

#include <string>
#include <utility>

namespace cyten {

struct SectorHdf5
{
    Sector sector;

    SectorHdf5() = default;
    explicit SectorHdf5(Sector s)
      : sector(std::move(s))
    {
    }

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
    {
        sector.save_hdf5(hdf5_saver, h5gr, subpath);
    }

    static SectorHdf5 from_hdf5(py::object hdf5_loader,
                                py::object h5gr,
                                std::string const& subpath)
    {
        return SectorHdf5{ Sector::from_hdf5(hdf5_loader, h5gr, subpath) };
    }
};

struct SectorArrayHdf5
{
    SectorArray sectors;

    SectorArrayHdf5() = default;
    explicit SectorArrayHdf5(SectorArray s)
      : sectors(std::move(s))
    {
    }

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
    {
        sectors.save_hdf5(hdf5_saver, h5gr, subpath);
    }

    static SectorArrayHdf5 from_hdf5(py::object hdf5_loader,
                                     py::object h5gr,
                                     std::string const& subpath)
    {
        return SectorArrayHdf5{ SectorArray::from_hdf5(hdf5_loader, h5gr, subpath) };
    }
};

} // namespace cyten
