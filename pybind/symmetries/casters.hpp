#pragma once

/// Optional conversion helpers: accept NumPy arrays / sequences as Sector arguments.
///
/// ``py::class_<Sector>`` / ``SectorArray`` own the Python types. These casters only
/// extend *load* so call sites can still pass ndarrays during migration; *cast*
/// always produces bound Sector / SectorArray instances (never bare ndarrays).

#include <cyten/symmetries/sector.h>
#include <cyten/symmetries/sector_numpy.h>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

namespace pybind11::detail {

template<>
struct type_caster<cyten::Sector> : type_caster_base<cyten::Sector>
{
    using base = type_caster_base<cyten::Sector>;

    bool load(handle src, bool convert)
    {
        if (base::load(src, convert)) {
            return true;
        }
        if (!convert) {
            return false;
        }
        try {
            cyten::Sector tmp = cyten::sector_from_numpy(src);
            value = new cyten::Sector(std::move(tmp));
            return true;
        } catch (...) {
            return false;
        }
    }
};

template<>
struct type_caster<cyten::SectorArray> : type_caster_base<cyten::SectorArray>
{
    using base = type_caster_base<cyten::SectorArray>;

    bool load(handle src, bool convert)
    {
        if (base::load(src, convert)) {
            return true;
        }
        if (!convert) {
            return false;
        }
        try {
            cyten::SectorArray tmp = cyten::sector_array_from_numpy(src);
            value = new cyten::SectorArray(std::move(tmp));
            return true;
        } catch (...) {
            return false;
        }
    }
};

} // namespace pybind11::detail
