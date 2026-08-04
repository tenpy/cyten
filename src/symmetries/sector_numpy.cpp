#include <cyten/symmetries/sector_numpy.h>

#include <array>
#include <limits>
#include <stdexcept>

namespace cyten {

namespace {

bool
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
load_sector_from_buffer(py::buffer_info const& info, Sector& out)
{
    if (info.ndim != 1) {
        return false;
    }
    auto const n = static_cast<std::size_t>(info.shape[0]);
    if (n > max_sector_ind_len) {
        return false;
    }
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    std::array<std::int16_t, max_sector_ind_len> buf{};
    for (std::size_t i = 0; i < n; ++i) {
        auto const v = static_cast<std::int64_t>(ptr[static_cast<ssize_t>(i) * stride]);
        if (!narrow_to_int16(v, buf[i])) {
            return false;
        }
    }
    out = Sector::from_span(std::span<const std::int16_t>(buf.data(), n));
    return true;
}

template<typename T>
bool
load_sector_array_from_buffer(py::buffer_info const& info, SectorArray& out)
{
    if (info.ndim != 2) {
        return false;
    }
    auto const num_sectors = static_cast<std::size_t>(info.shape[0]);
    auto const sector_ind_len = static_cast<std::size_t>(info.shape[1]);
    if (sector_ind_len > max_sector_ind_len) {
        return false;
    }
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride0 = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    auto const stride1 = info.strides[1] / static_cast<ssize_t>(sizeof(T));
    out = SectorArray(num_sectors, static_cast<std::uint8_t>(sector_ind_len));
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

} // namespace

py::array
sector_to_numpy(Sector const& src)
{
    py::array_t<std::int64_t> arr(static_cast<ssize_t>(src.len()));
    auto r = arr.mutable_unchecked<1>();
    for (std::uint8_t i = 0; i < src.len(); ++i) {
        r(i) = src.q[i];
    }
    return arr;
}

py::array
sector_array_to_numpy(SectorArray const& src)
{
    py::array_t<std::int64_t> arr(
      { static_cast<ssize_t>(src.num_sectors), static_cast<ssize_t>(src.sector_ind_len) });
    auto r = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < src.num_sectors; ++i) {
        for (std::uint8_t j = 0; j < src.sector_ind_len; ++j) {
            r(static_cast<ssize_t>(i), static_cast<ssize_t>(j)) =
              src.data[i * src.sector_ind_len + j];
        }
    }
    return arr;
}

Sector
sector_from_numpy(py::handle src)
{
    Sector out;
    py::array arr = py::array::ensure(src);
    if (!arr) {
        throw std::invalid_argument("sector_from_numpy: expected array-like");
    }
    auto const info = arr.request();
    bool ok = false;
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        ok = load_sector_from_buffer<std::int16_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int32_t>()) {
        ok = load_sector_from_buffer<std::int32_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int64_t>()) {
        ok = load_sector_from_buffer<std::int64_t>(info, out);
    } else {
        auto casted =
          py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(src);
        if (casted) {
            ok = load_sector_from_buffer<std::int64_t>(casted.request(), out);
        }
    }
    if (!ok) {
        throw std::invalid_argument("sector_from_numpy: invalid sector array");
    }
    return out;
}

SectorArray
sector_array_from_numpy(py::handle src)
{
    SectorArray out;
    py::array arr = py::array::ensure(src);
    if (!arr) {
        throw std::invalid_argument("sector_array_from_numpy: expected array-like");
    }
    auto const info = arr.request();
    bool ok = false;
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        ok = load_sector_array_from_buffer<std::int16_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int32_t>()) {
        ok = load_sector_array_from_buffer<std::int32_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int64_t>()) {
        ok = load_sector_array_from_buffer<std::int64_t>(info, out);
    } else {
        auto casted =
          py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(src);
        if (casted) {
            ok = load_sector_array_from_buffer<std::int64_t>(casted.request(), out);
        }
    }
    if (!ok) {
        throw std::invalid_argument("sector_array_from_numpy: invalid sector array");
    }
    return out;
}

void
Sector::save_hdf5(py::object hdf5_saver, py::object /*h5gr*/, std::string const& subpath) const
{
    // ``subpath`` is the group already created by ``hdf5_saver``; store charges under ``values``.
    hdf5_saver.attr("save")(sector_to_numpy(*this), subpath + "values");
}

Sector
Sector::from_hdf5(py::object hdf5_loader, py::object /*h5gr*/, std::string const& subpath)
{
    return sector_from_numpy(hdf5_loader.attr("load")(subpath + "values"));
}

void
SectorArray::save_hdf5(py::object hdf5_saver,
                       py::object /*h5gr*/,
                       std::string const& subpath) const
{
    hdf5_saver.attr("save")(sector_array_to_numpy(*this), subpath + "values");
}

SectorArray
SectorArray::from_hdf5(py::object hdf5_loader, py::object /*h5gr*/, std::string const& subpath)
{
    return sector_array_from_numpy(hdf5_loader.attr("load")(subpath + "values"));
}

py::object
sector_as_hdf5_exportable(Sector const& src)
{
    // Bound ``SectorHdf5`` as ``cyten._core.Sector`` (not ``Sector`` itself — casters).
    return py::module_::import("cyten._core").attr("Sector")(sector_to_numpy(src));
}

py::object
sector_array_as_hdf5_exportable(SectorArray const& src)
{
    return py::module_::import("cyten._core").attr("SectorArray")(sector_array_to_numpy(src));
}

Sector
sector_from_hdf5_object(py::handle src)
{
    // Bound Sector from HDF5, or a plain ndarray (legacy / caster path).
    if (py::hasattr(src, "__array__") && !py::isinstance<py::array>(src)) {
        return sector_from_numpy(src.attr("__array__")());
    }
    return sector_from_numpy(src);
}

SectorArray
sector_array_from_hdf5_object(py::handle src)
{
    if (py::hasattr(src, "__array__") && !py::isinstance<py::array>(src)) {
        return sector_array_from_numpy(src.attr("__array__")());
    }
    return sector_array_from_numpy(src);
}

} // namespace cyten
