#pragma once

/// Optional conversion helpers: accept NumPy arrays / sequences as Sector arguments.
///
/// ``py::class_<Sector>`` / ``SectorArray`` own the Python types. These casters only
/// extend *load* so call sites can still pass ndarrays during migration; *cast*
/// always produces bound Sector / SectorArray instances (never bare ndarrays).

#include <cyten/symmetries/fusion_symbol.h>
#include <cyten/symmetries/sector.h>
#include <cyten/symmetries/sector_numpy.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace cyten {

inline py::array
vector_i64_to_numpy(std::vector<int64> const& v)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(v.size()));
    auto* ptr = arr.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return arr;
}

inline py::array
vector_f64_to_numpy(std::vector<float64> const& v)
{
    py::array_t<float64> arr(static_cast<py::ssize_t>(v.size()));
    auto* ptr = arr.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return arr;
}

inline std::vector<int64>
vector_i64_from_numpy(py::object obj)
{
    auto arr = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error("expected 1D int array");
    }
    auto info = arr.request();
    if (info.ndim != 1) {
        throw py::type_error("expected 1D int array");
    }
    auto const* ptr = static_cast<int64 const*>(info.ptr);
    return std::vector<int64>(ptr, ptr + info.shape[0]);
}

inline std::vector<float64>
vector_f64_from_numpy(py::object obj)
{
    auto arr = py::array_t<float64, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error("expected 1D float array");
    }
    auto info = arr.request();
    if (info.ndim != 1) {
        throw py::type_error("expected 1D float array");
    }
    auto const* ptr = static_cast<float64 const*>(info.ptr);
    return std::vector<float64>(ptr, ptr + info.shape[0]);
}

} // namespace cyten

namespace pybind11::detail {

/// Transparent FusionSymbol ↔ numpy.ndarray for bindings / trampolines.
template<>
struct type_caster<cyten::FusionSymbol>
{
    PYBIND11_TYPE_CASTER(cyten::FusionSymbol, const_name("numpy.ndarray"));

    bool load(handle src, bool convert)
    {
        if (!convert && !isinstance<array>(src)) {
            return false;
        }
        try {
            value = cyten::fusion_symbol_from_numpy(array::ensure(src));
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle cast(cyten::FusionSymbol const& src,
                       return_value_policy /*policy*/,
                       handle /*parent*/)
    {
        return cyten::fusion_symbol_to_numpy(src).release();
    }
};

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
