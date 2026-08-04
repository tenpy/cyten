#pragma once

/// Type casters: cyten::Sector / SectorArray ↔ NumPy ndarrays.
///
/// Do not bind Sector as a py::class_. Include this header from binding TUs that
/// cross the Python boundary with Sector or SectorArray.

#include <cyten/symmetries/sector.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>

namespace pybind11::detail {

namespace cyten_sector_casters_detail {

inline bool
narrow_to_int16(std::int64_t v, std::int16_t& out)
{
    if (v < std::numeric_limits<std::int16_t>::min() ||
        v > std::numeric_limits<std::int16_t>::max()) {
        return false;
    }
    out = static_cast<std::int16_t>(v);
    return true;
}

template<typename T>
bool
load_sector_from_buffer(buffer_info const& info, cyten::Sector& out)
{
    if (info.ndim != 1) {
        return false;
    }
    auto const n = static_cast<std::size_t>(info.shape[0]);
    if (n > cyten::max_sector_ind_len) {
        return false;
    }
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    std::array<std::int16_t, cyten::max_sector_ind_len> buf{};
    for (std::size_t i = 0; i < n; ++i) {
        auto const v = static_cast<std::int64_t>(ptr[static_cast<ssize_t>(i) * stride]);
        if (!narrow_to_int16(v, buf[i])) {
            return false;
        }
    }
    out = cyten::Sector::from_span(std::span<const std::int16_t>(buf.data(), n));
    return true;
}

template<typename T>
bool
load_sector_array_from_buffer(buffer_info const& info, cyten::SectorArray& out)
{
    if (info.ndim != 2) {
        return false;
    }
    auto const num_sectors = static_cast<std::size_t>(info.shape[0]);
    auto const sector_ind_len = static_cast<std::size_t>(info.shape[1]);
    if (sector_ind_len > cyten::max_sector_ind_len) {
        return false;
    }
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride0 = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    auto const stride1 = info.strides[1] / static_cast<ssize_t>(sizeof(T));
    out = cyten::SectorArray(num_sectors, static_cast<std::uint8_t>(sector_ind_len));
    for (std::size_t i = 0; i < num_sectors; ++i) {
        for (std::size_t j = 0; j < sector_ind_len; ++j) {
            auto const v = static_cast<std::int64_t>(
              ptr[static_cast<ssize_t>(i) * stride0 + static_cast<ssize_t>(j) * stride1]);
            std::int16_t narrow = 0;
            if (!narrow_to_int16(v, narrow)) {
                return false;
            }
            out.data[i * sector_ind_len + j] = narrow;
        }
    }
    return true;
}

inline bool
load_sector(handle src, bool convert, cyten::Sector& out)
{
    if (!src) {
        return false;
    }
    array arr = array::ensure(src);
    if (!arr) {
        return false;
    }
    if (!convert && !isinstance<array>(src)) {
        return false;
    }
    auto const info = arr.request();
    if (info.ndim != 1) {
        return false;
    }
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        return load_sector_from_buffer<std::int16_t>(info, out);
    }
    if (info.item_type_is_equivalent_to<std::int32_t>()) {
        return load_sector_from_buffer<std::int32_t>(info, out);
    }
    if (info.item_type_is_equivalent_to<std::int64_t>()) {
        return load_sector_from_buffer<std::int64_t>(info, out);
    }
    // Force-cast to int64 then narrow (covers Python int lists via ensure + dtype).
    auto casted = array_t<std::int64_t, array::c_style | array::forcecast>::ensure(src);
    if (!casted) {
        return false;
    }
    return load_sector_from_buffer<std::int64_t>(casted.request(), out);
}

inline bool
load_sector_array(handle src, bool convert, cyten::SectorArray& out)
{
    if (!src) {
        return false;
    }
    array arr = array::ensure(src);
    if (!arr) {
        return false;
    }
    if (!convert && !isinstance<array>(src)) {
        return false;
    }
    auto const info = arr.request();
    if (info.ndim != 2) {
        return false;
    }
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        return load_sector_array_from_buffer<std::int16_t>(info, out);
    }
    if (info.item_type_is_equivalent_to<std::int32_t>()) {
        return load_sector_array_from_buffer<std::int32_t>(info, out);
    }
    if (info.item_type_is_equivalent_to<std::int64_t>()) {
        return load_sector_array_from_buffer<std::int64_t>(info, out);
    }
    auto casted = array_t<std::int64_t, array::c_style | array::forcecast>::ensure(src);
    if (!casted) {
        return false;
    }
    return load_sector_array_from_buffer<std::int64_t>(casted.request(), out);
}

inline handle
cast_sector(cyten::Sector const& src)
{
    array_t<std::int64_t> arr(static_cast<ssize_t>(src.len()));
    auto r = arr.mutable_unchecked<1>();
    for (std::uint8_t i = 0; i < src.len(); ++i) {
        r(i) = src.q[i];
    }
    return arr.release();
}

inline handle
cast_sector_array(cyten::SectorArray const& src)
{
    array_t<std::int64_t> arr(
      { static_cast<ssize_t>(src.num_sectors), static_cast<ssize_t>(src.sector_ind_len) });
    auto r = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < src.num_sectors; ++i) {
        for (std::uint8_t j = 0; j < src.sector_ind_len; ++j) {
            r(static_cast<ssize_t>(i), static_cast<ssize_t>(j)) =
              src.data[i * src.sector_ind_len + j];
        }
    }
    return arr.release();
}

} // namespace cyten_sector_casters_detail

template<>
struct type_caster<cyten::Sector>
{
    PYBIND11_TYPE_CASTER(cyten::Sector, const_name("numpy.ndarray"));

    bool load(handle src, bool convert)
    {
        return cyten_sector_casters_detail::load_sector(src, convert, value);
    }

    static handle cast(cyten::Sector const& src,
                       return_value_policy /* policy */,
                       handle /* parent */)
    {
        return cyten_sector_casters_detail::cast_sector(src);
    }
};

template<>
struct type_caster<cyten::SectorArray>
{
    PYBIND11_TYPE_CASTER(cyten::SectorArray, const_name("numpy.ndarray"));

    bool load(handle src, bool convert)
    {
        return cyten_sector_casters_detail::load_sector_array(src, convert, value);
    }

    static handle cast(cyten::SectorArray const& src,
                       return_value_policy /* policy */,
                       handle /* parent */)
    {
        return cyten_sector_casters_detail::cast_sector_array(src);
    }
};

} // namespace pybind11::detail
